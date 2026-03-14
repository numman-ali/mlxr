import AppKit
import MLXRDesignSystem
import SwiftUI

struct ImageCanvasPreview: View {
    let url: URL

    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                MediaHero(image: image, dominantHue: dominantHue(from: image))
            } else {
                EmptyStateView(
                    title: "Loading preview",
                    subtitle: "Preparing the selected image for display.",
                    systemImage: "photo"
                )
            }
        }
        .task(id: url) {
            image = await Task.detached(priority: .userInitiated) {
                NSImage(contentsOf: url)
            }
            .value
        }
    }
}
