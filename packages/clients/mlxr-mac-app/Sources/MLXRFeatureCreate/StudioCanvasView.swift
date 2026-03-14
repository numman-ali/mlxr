import AVKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct StudioCanvasView: View {
    let task: ProductTask
    let selectedAsset: LibraryAsset?
    let selectedAssetURL: URL?
    let referenceAssets: [LibraryAsset]
    let currentJob: JobRecord?
    let currentJobPhase: String?
    let currentRunGroupTitle: String?
    let currentRunGroupAssets: [LibraryAsset]
    @Binding var focusedResultAssetId: String?
    let planningError: String?
    let onDismissPlanningError: () -> Void
    let error: String?
    let onDismissError: () -> Void
    let onUseFocusedAssetAsReference: () -> Void
    let onEditAsset: (LibraryAsset) -> Void
    let onAnimateAsset: (LibraryAsset) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
            if let planningError {
                InlineErrorBanner(planningError, onDismiss: onDismissPlanningError)
            }
            if let error {
                InlineErrorBanner(error, onDismiss: onDismissError)
            }

            GlassCard(
                title: canvasTitle,
                subtitle: canvasSubtitle
            ) {
                previewArea
                    .frame(maxWidth: .infinity, minHeight: 380, maxHeight: .infinity)

                HStack(spacing: MLXRSpacing.xs) {
                    if let selectedAsset {
                        StatusPill(label: selectedAsset.sourceSummary, tint: MLXRColor.brandSecondary)
                    }
                    if !referenceAssets.isEmpty {
                        StatusPill(label: "\(referenceAssets.count) reference\(referenceAssets.count == 1 ? "" : "s")", tint: MLXRColor.brandPrimary)
                    }
                    if !currentRunGroupAssets.isEmpty {
                        StatusPill(label: "\(currentRunGroupAssets.count) result\(currentRunGroupAssets.count == 1 ? "" : "s")", tint: MLXRColor.brandWarm)
                    }
                    Spacer()
                    if selectedAsset != nil {
                        Button("Use as reference", action: onUseFocusedAssetAsReference)
                            .buttonStyle(.bordered)
                    }
                }

                if !currentRunGroupAssets.isEmpty {
                    resultFilmstrip
                }
            }
        }
        .padding(.horizontal, MLXRSpacing.lg)
        .padding(.vertical, MLXRSpacing.xl)
    }

    private var canvasTitle: String {
        currentRunGroupTitle ?? task.title
    }

    private var canvasSubtitle: String {
        if let currentRunGroupTitle, !currentRunGroupTitle.isEmpty {
            return "Your current run stays here so you don’t need to hunt through a separate jobs page."
        }
        return task.subtitle
    }

    @ViewBuilder
    private var previewArea: some View {
        if let currentJob, !currentJob.state.isTerminal {
            VStack(spacing: MLXRSpacing.md) {
                MLXRProgressRing(progress: progress(for: currentJob))
                Text(currentJobPhase ?? "Running")
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)
                if let promptValue = currentJob.request.inputs.string("prompt"), !promptValue.isEmpty {
                    Text(promptValue)
                        .font(MLXRType.bodyLarge)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .multilineTextAlignment(.center)
                        .lineLimit(4)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let selectedAssetURL, let selectedAsset {
            VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                StudioPreviewView(url: selectedAssetURL, asset: selectedAsset)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                actionRow(for: selectedAsset)
            }
        } else {
            EmptyStateView(
                title: "Pick something to work on",
                subtitle: "Start a fresh run, or select an existing asset on the left to edit, animate, or reuse.",
                systemImage: "sparkles.rectangle.stack.fill"
            )
        }
    }

    private var resultFilmstrip: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text("Current results")
                .font(MLXRType.bodySmall)
                .foregroundStyle(MLXRColor.textSecondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: MLXRSpacing.sm) {
                    ForEach(currentRunGroupAssets) { asset in
                        Button {
                            focusedResultAssetId = asset.id
                        } label: {
                            VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                                ThumbnailTile(asset: asset, isSelected: focusedResultAssetId == asset.id)
                                    .frame(width: 136, height: 96)
                                Text(asset.displayTitle)
                                    .font(MLXRType.captionLarge)
                                    .foregroundStyle(MLXRColor.textPrimary)
                                    .lineLimit(1)
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func actionRow(for asset: LibraryAsset) -> some View {
        HStack(spacing: MLXRSpacing.sm) {
            Button("Use as reference", action: onUseFocusedAssetAsReference)
                .buttonStyle(.borderedProminent)

            if asset.isImage {
                Button("Edit this") { onEditAsset(asset) }
                    .buttonStyle(.bordered)
                Button("Animate this") { onAnimateAsset(asset) }
                    .buttonStyle(.bordered)
            }
        }
    }

    private func progress(for job: JobRecord) -> Double {
        switch job.state {
        case .accepted: 0.05
        case .preparing: 0.10
        case .loadingModel: 0.18
        case .running: 0.55
        case .streamingOutput: 0.88
        case .finalizing: 0.96
        default: 0.25
        }
    }
}

private struct StudioPreviewView: View {
    let url: URL
    let asset: LibraryAsset

    var body: some View {
        Group {
            if asset.isImage {
                ImageCanvasPreview(url: url)
            } else if asset.isVideo {
                VideoCanvasPreview(url: url)
            } else {
                EmptyStateView(
                    title: asset.displayTitle,
                    subtitle: asset.subtitle,
                    systemImage: "doc"
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct ThumbnailTile: View {
    let asset: LibraryAsset
    let isSelected: Bool

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            isSelected ? MLXRColor.brandPrimary.opacity(0.45) : MLXRColor.surfaceCard,
                            MLXRColor.canvasRaised,
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            Image(systemName: icon)
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(isSelected ? MLXRColor.textPrimary : MLXRColor.textSecondary)
        }
        .overlay(
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .strokeBorder(isSelected ? MLXRColor.brandPrimary.opacity(0.8) : MLXRColor.borderSubtle, lineWidth: 1)
        )
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

private struct VideoCanvasPreview: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        view.videoGravity = .resizeAspect
        view.player = AVPlayer(url: url)
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        let currentURL = (nsView.player?.currentItem?.asset as? AVURLAsset)?.url
        if currentURL != url {
            nsView.player = AVPlayer(url: url)
        }
    }

    static func dismantleNSView(_ nsView: AVPlayerView, coordinator: ()) {
        nsView.player?.pause()
        nsView.player = nil
    }
}
