import AppKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct LibraryAssetThumbnailView: View {
    let asset: LibraryAsset
    let onMaterialize: @Sendable (LibraryAsset) async -> URL?

    @State private var previewImage: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(MLXRColor.surfaceCard)

            if let previewImage {
                Image(nsImage: previewImage)
                    .resizable()
                    .scaledToFill()
            } else {
                VStack(spacing: MLXRSpacing.xs) {
                    Image(systemName: icon)
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(MLXRColor.textSecondary)
                    Text(asset.kind.rawValue.capitalized)
                        .font(MLXRType.captionSmall)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
        .task(id: asset.id) {
            previewImage = await LibraryThumbnailStore.shared.thumbnail(
                for: asset,
                maxPixelSize: 640,
                materialize: onMaterialize
            )
        }
    }

    private var icon: String {
        switch asset.kind {
        case .image: "photo.fill"
        case .video: "film.fill"
        case .audio: "waveform"
        case .other: "doc.fill"
        }
    }
}
