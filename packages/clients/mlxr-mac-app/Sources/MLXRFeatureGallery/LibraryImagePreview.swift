import MLXRDesignSystem
import SwiftUI

struct LibraryImagePreview: View {
    let url: URL

    @State private var image: NSImage?
    @State private var dominantHueValue: CGFloat?

    var body: some View {
        Group {
            if let image {
                MediaHero(image: image, dominantHue: dominantHueValue)
            } else {
                EmptyStateView(
                    title: "Loading preview",
                    subtitle: "Preparing the selected image for display.",
                    systemImage: "photo"
                )
            }
        }
        .task(id: url) {
            image = nil
            dominantHueValue = nil
            let loadedPreview = await PreviewImageLoader.load(from: url)
            guard !Task.isCancelled else {
                return
            }
            image = loadedPreview.0
            dominantHueValue = loadedPreview.1
        }
    }
}
