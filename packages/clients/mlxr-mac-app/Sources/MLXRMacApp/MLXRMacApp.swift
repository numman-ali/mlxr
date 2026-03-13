import MLXRAppShell
import SwiftUI

@main
struct MLXRMacApp: App {
    var body: some Scene {
        WindowGroup {
            MLXRMacAppRoot()
                .frame(minWidth: 1120, minHeight: 760)
        }
        .windowResizability(.contentSize)
    }
}
