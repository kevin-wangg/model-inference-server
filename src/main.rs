mod inference;
mod server;

use crate::server::Server;

#[tokio::main]
async fn main() {
    let server = Server {};
    server.run().await;
}
