#[derive(Clone)]
pub struct AppState {
    app_name: String
}

impl AppState {
    pub fn new(app_name: &str) -> Self {
        Self {
            app_name: app_name.into()
        }
    }

    pub fn get_name(&self) -> String {
        self.app_name.clone()
    }
}


