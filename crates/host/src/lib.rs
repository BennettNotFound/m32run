//! Host support functions
//!
//! This crate is reserved for functions that interface directly with
//! the host operating system.  In a more complete implementation
//! these would include file descriptors, timers, network sockets and
//! other resources needed to satisfy guest syscalls.  At present this
//! module is empty, serving only as a placeholder to demonstrate
//! project structure.

use std::path::Path;

#[cfg(target_os = "macos")]
pub fn prepare_macos_ui_app() -> Result<(), String> {
    use cocoa::appkit::NSApp;
    use cocoa::base::id;
    use objc::{class, msg_send, sel, sel_impl};

    unsafe {
        let mut app: id = NSApp();
        if app.is_null() {
            app = msg_send![class!(NSApplication), sharedApplication];
        }
        if app.is_null() {
            return Err("failed to acquire NSApplication".to_string());
        }
        // NSApplicationActivationPolicyRegular = 0
        let _: () = msg_send![app, setActivationPolicy: 0isize];
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub fn prepare_macos_ui_app() -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "macos")]
pub fn set_dock_icon_from_file(path: &Path) -> Result<(), String> {
    use cocoa::appkit::NSApp;
    use cocoa::base::{id, nil};
    use cocoa::foundation::{NSAutoreleasePool, NSString};
    use objc::{class, msg_send, sel, sel_impl};

    let path_text = path.to_string_lossy();
    if path_text.is_empty() {
        return Err("dock icon path is empty".to_string());
    }

    unsafe {
        let pool = NSAutoreleasePool::new(nil);
        let mut app: id = NSApp();
        if app.is_null() {
            app = msg_send![class!(NSApplication), sharedApplication];
        }
        if app.is_null() {
            let _: () = msg_send![pool, drain];
            return Err("failed to acquire NSApplication".to_string());
        }

        let ns_path = NSString::alloc(nil).init_str(&path_text);
        let image: id = msg_send![class!(NSImage), alloc];
        let image: id = msg_send![image, initWithContentsOfFile: ns_path];
        if image.is_null() {
            let _: () = msg_send![pool, drain];
            return Err(format!(
                "failed to load icon image from '{}'",
                path.display()
            ));
        }

        let _: () = msg_send![app, setApplicationIconImage: image];
        let _: () = msg_send![pool, drain];
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_dock_icon_from_file(_path: &Path) -> Result<(), String> {
    Ok(())
}
