//! 图形后端抽象。
//!
//! 目标是给 guest 提供一个“真实可见”的宿主图形闭环：
//! - 可创建窗口
//! - 可创建并更新像素帧
//! - 可轮询事件避免窗口消息饿死
//! - 可（尽力）探测 OpenGL 兼容上下文是否可用

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsCaps {
    pub context_created: bool,
}

#[derive(Debug, Clone)]
pub enum HostInputEvent {
    Quit,
    KeyDown { keycode: i32 },
    KeyUp { keycode: i32 },
    MouseMove { x: i32, y: i32 },
    MouseDown { button: u8, x: i32, y: i32 },
    MouseUp { button: u8, x: i32, y: i32 },
    MouseWheel { x: i32, y: i32 },
    TextInput { text: String },
}

pub trait GraphicsBackend {
    fn backend_name(&self) -> &'static str;
    fn ensure_window(&mut self, title: &str, width: u32, height: u32) -> Result<(), String>;
    fn set_title(&mut self, title: &str) -> Result<(), String>;
    fn resize(&mut self, width: u32, height: u32) -> Result<(), String>;
    fn present_rgb(&mut self, rgb: &[u32], width: u32, height: u32) -> Result<(), String>;
    fn poll_events(&mut self) -> Result<(), String>;
    fn drain_events(&mut self) -> Vec<HostInputEvent>;
    fn clipboard_get(&self) -> Option<String>;
    fn clipboard_set(&mut self, text: &str) -> Result<(), String>;
    fn is_open(&self) -> bool;
    fn caps(&self) -> GraphicsCaps;
}

#[cfg(target_os = "macos")]
pub fn create_default_graphics_backend() -> Result<Box<dyn GraphicsBackend>, String> {
    Ok(Box::new(Sdl2GraphicsBackend::new()?))
}

#[cfg(not(target_os = "macos"))]
pub fn create_default_graphics_backend() -> Result<Box<dyn GraphicsBackend>, String> {
    Err("graphics backend is currently implemented for macOS only".to_string())
}

#[cfg(target_os = "macos")]
struct Sdl2GraphicsBackend {
    sdl: sdl2::Sdl,
    video: sdl2::VideoSubsystem,
    event_pump: sdl2::EventPump,
    canvas: Option<sdl2::render::Canvas<sdl2::video::Window>>,
    width: u32,
    height: u32,
    window_open: bool,
    context_created: bool,
    pending_events: Vec<HostInputEvent>,
    _gl_probe_window: Option<sdl2::video::Window>,
    _gl_probe_context: Option<sdl2::video::GLContext>,
}

#[cfg(target_os = "macos")]
impl Sdl2GraphicsBackend {
    fn new() -> Result<Self, String> {
        let sdl = sdl2::init().map_err(|e| format!("sdl init failed: {e}"))?;
        let video = sdl
            .video()
            .map_err(|e| format!("sdl video init failed: {e}"))?;

        // 先做一次 OpenGL 兼容上下文探测：
        // 这能告诉我们宿主上是否可建立 GL 上下文（即使当前首版先走像素上传路径）。
        let gl_attr = video.gl_attr();
        gl_attr.set_context_profile(sdl2::video::GLProfile::Compatibility);
        gl_attr.set_context_version(2, 1);

        let mut context_created = false;
        let mut gl_probe_window = None;
        let mut gl_probe_context = None;

        // TODO: 后续把 guest 的 OpenGL 调用真正映射到这个 context，
        // 目前首版先保证窗口 + 像素帧 present 可用，避免初始化阶段卡死。
        if let Ok(window) = video
            .window("m32run-gl-probe", 16, 16)
            .hidden()
            .opengl()
            .build()
        {
            let probe_window = window;
            match probe_window.gl_create_context() {
                Ok(ctx) => {
                    context_created = true;
                    gl_probe_context = Some(ctx);
                    gl_probe_window = Some(probe_window);
                }
                Err(_) => {
                    // 探测失败不阻止主窗口路径。
                }
            }
        }

        let event_pump = sdl
            .event_pump()
            .map_err(|e| format!("sdl event pump init failed: {e}"))?;

        Ok(Self {
            sdl,
            video,
            event_pump,
            canvas: None,
            width: 0,
            height: 0,
            window_open: true,
            context_created,
            pending_events: Vec::new(),
            _gl_probe_window: gl_probe_window,
            _gl_probe_context: gl_probe_context,
        })
    }

    fn create_canvas(
        &self,
        title: &str,
        width: u32,
        height: u32,
    ) -> Result<sdl2::render::Canvas<sdl2::video::Window>, String> {
        let make_window = || {
            self.video
                .window(title, width.max(1), height.max(1))
                .position_centered()
                .resizable()
                .allow_highdpi()
                .build()
                .map_err(|e| format!("window create failed: {e}"))
        };

        let window = make_window()?;
        match window.into_canvas().accelerated().present_vsync().build() {
            Ok(canvas) => Ok(canvas),
            Err(_err) => {
                // 加速渲染器不可用时回退到软件路径，保证“可显示”优先。
                let window = make_window()?;
                window
                    .into_canvas()
                    .software()
                    .build()
                    .map_err(|e| format!("software renderer create failed: {e}"))
            }
        }
    }

    fn ensure_canvas(&mut self, title: &str, width: u32, height: u32) -> Result<(), String> {
        if self.canvas.is_none() {
            let mut canvas = self.create_canvas(title, width, height)?;
            canvas.window_mut().show();
            canvas.window_mut().raise();
            canvas.clear();
            canvas.present();
            self.width = width.max(1);
            self.height = height.max(1);
            self.canvas = Some(canvas);
            self.window_open = true;
            return Ok(());
        }
        self.resize(width, height)?;
        self.set_title(title)?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl GraphicsBackend for Sdl2GraphicsBackend {
    fn backend_name(&self) -> &'static str {
        "sdl2"
    }

    fn ensure_window(&mut self, title: &str, width: u32, height: u32) -> Result<(), String> {
        self.ensure_canvas(title, width, height)
    }

    fn set_title(&mut self, title: &str) -> Result<(), String> {
        if let Some(canvas) = self.canvas.as_mut() {
            canvas
                .window_mut()
                .set_title(title)
                .map_err(|e| format!("set window title failed: {e}"))?;
        }
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;
        if let Some(canvas) = self.canvas.as_mut() {
            canvas
                .window_mut()
                .set_size(width, height)
                .map_err(|e| format!("resize window failed: {e}"))?;
        }
        Ok(())
    }

    fn present_rgb(&mut self, rgb: &[u32], width: u32, height: u32) -> Result<(), String> {
        if !self.window_open {
            return Ok(());
        }
        if width == 0 || height == 0 {
            return Ok(());
        }
        let need = (width as usize).saturating_mul(height as usize);
        if rgb.len() < need {
            return Err(format!(
                "rgb buffer too small: got {}, need {}",
                rgb.len(),
                need
            ));
        }

        self.resize(width, height)?;
        let Some(canvas) = self.canvas.as_mut() else {
            return Ok(());
        };
        let creator = canvas.texture_creator();
        let mut texture = creator
            .create_texture_streaming(sdl2::pixels::PixelFormatEnum::ARGB8888, width, height)
            .map_err(|e| format!("create texture failed: {e}"))?;

        texture
            .with_lock(None, |buf, pitch| {
                let row_bytes = (width as usize) * 4;
                for y in 0..(height as usize) {
                    let src_row = &rgb[y * (width as usize)..(y + 1) * (width as usize)];
                    let dst = &mut buf[y * pitch..y * pitch + row_bytes];
                    for (x, src_pixel) in src_row.iter().enumerate() {
                        let argb = 0xff00_0000u32 | *src_pixel;
                        let off = x * 4;
                        dst[off..off + 4].copy_from_slice(&argb.to_ne_bytes());
                    }
                }
            })
            .map_err(|e| format!("lock texture failed: {e}"))?;

        canvas.clear();
        canvas
            .copy(&texture, None, None)
            .map_err(|e| format!("copy texture failed: {e}"))?;
        canvas.present();
        Ok(())
    }

    fn poll_events(&mut self) -> Result<(), String> {
        use sdl2::event::Event;
        use sdl2::mouse::MouseButton;
        use sdl2::event::WindowEvent;

        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    self.window_open = false;
                    self.pending_events.push(HostInputEvent::Quit);
                }
                Event::Window {
                    win_event: WindowEvent::Close,
                    ..
                } => {
                    self.window_open = false;
                    self.pending_events.push(HostInputEvent::Quit);
                }
                Event::MouseMotion { x, y, .. } => {
                    self.pending_events.push(HostInputEvent::MouseMove { x, y });
                }
                Event::MouseButtonDown {
                    mouse_btn, x, y, ..
                } => {
                    let button = match mouse_btn {
                        MouseButton::Left => 1,
                        MouseButton::Right => 2,
                        MouseButton::Middle => 3,
                        MouseButton::X1 => 4,
                        MouseButton::X2 => 5,
                        MouseButton::Unknown => 0,
                    };
                    self.pending_events
                        .push(HostInputEvent::MouseDown { button, x, y });
                }
                Event::MouseButtonUp {
                    mouse_btn, x, y, ..
                } => {
                    let button = match mouse_btn {
                        MouseButton::Left => 1,
                        MouseButton::Right => 2,
                        MouseButton::Middle => 3,
                        MouseButton::X1 => 4,
                        MouseButton::X2 => 5,
                        MouseButton::Unknown => 0,
                    };
                    self.pending_events
                        .push(HostInputEvent::MouseUp { button, x, y });
                }
                Event::MouseWheel { x, y, .. } => {
                    self.pending_events
                        .push(HostInputEvent::MouseWheel { x, y });
                }
                Event::KeyDown { keycode, .. } => {
                    if let Some(code) = keycode {
                        self.pending_events
                            .push(HostInputEvent::KeyDown {
                                keycode: i32::from(code),
                            });
                    }
                }

                Event::KeyUp { keycode, .. } => {
                    if let Some(code) = keycode {
                        self.pending_events
                            .push(HostInputEvent::KeyUp {
                                keycode: i32::from(code),
                            });
                    }
                }
                Event::TextInput { text, .. } => {
                    if !text.is_empty() {
                        self.pending_events
                            .push(HostInputEvent::TextInput { text });
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn drain_events(&mut self) -> Vec<HostInputEvent> {
        std::mem::take(&mut self.pending_events)
    }

    fn clipboard_get(&self) -> Option<String> {
        self.video.clipboard().clipboard_text().ok()
    }

    fn clipboard_set(&mut self, text: &str) -> Result<(), String> {
        self.video
            .clipboard()
            .set_clipboard_text(text)
            .map_err(|e| format!("set clipboard text failed: {e}"))
    }

    fn is_open(&self) -> bool {
        self.window_open
    }

    fn caps(&self) -> GraphicsCaps {
        GraphicsCaps {
            context_created: self.context_created,
        }
    }
}
