import MLXRAppShell
import SwiftUI

@main
struct MLXRMacApp: App {
    var body: some Scene {
        WindowGroup("MLXR") {
            MLXRMacAppRoot()
                .frame(minWidth: 1200, minHeight: 800)
        }
        .windowResizability(.contentSize)
    }
}
