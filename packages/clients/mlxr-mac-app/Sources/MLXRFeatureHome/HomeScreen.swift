@preconcurrency import SwiftUI
import MLXRAppDomain
import MLXRDesignSystem

@MainActor
public struct HomeScreen: View {
    let runtimeReady: Bool
    let hasModels: Bool
    let hasContent: Bool
    let installedModelCount: Int
    let activeJobCount: Int
    let queuedInstallCount: Int
    let totalCreations: Int
    let recentAssets: [LibraryAsset]
    let hasWorkspaceDraft: Bool
    let onCreateImage: () -> Void
    let onCreateVideo: () -> Void
    let onOpenModels: () -> Void
    let onOpenLibrary: () -> Void
    let onContinueAsset: (LibraryAsset) -> Void
    let onResumeWorkspace: () -> Void

    public init(
        runtimeReady: Bool,
        hasModels: Bool,
        hasContent: Bool,
        installedModelCount: Int,
        activeJobCount: Int,
        queuedInstallCount: Int,
        totalCreations: Int,
        recentAssets: [LibraryAsset],
        hasWorkspaceDraft: Bool,
        onCreateImage: @escaping () -> Void,
        onCreateVideo: @escaping () -> Void,
        onOpenModels: @escaping () -> Void,
        onOpenLibrary: @escaping () -> Void,
        onContinueAsset: @escaping (LibraryAsset) -> Void,
        onResumeWorkspace: @escaping () -> Void
    ) {
        self.runtimeReady = runtimeReady
        self.hasModels = hasModels
        self.hasContent = hasContent
        self.installedModelCount = installedModelCount
        self.activeJobCount = activeJobCount
        self.queuedInstallCount = queuedInstallCount
        self.totalCreations = totalCreations
        self.recentAssets = recentAssets
        self.hasWorkspaceDraft = hasWorkspaceDraft
        self.onCreateImage = onCreateImage
        self.onCreateVideo = onCreateVideo
        self.onOpenModels = onOpenModels
        self.onOpenLibrary = onOpenLibrary
        self.onContinueAsset = onContinueAsset
        self.onResumeWorkspace = onResumeWorkspace
    }

    public var body: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                header
                quickActions

                if hasWorkspaceDraft {
                    resumeCard
                } else if !hasModels {
                    setupCard
                } else if !hasContent {
                    firstCreationCard
                } else {
                    continueCard
                }
            }
            .padding(.horizontal, MLXRSpacing.xl)
            .padding(.top, MLXRSpacing.xl)
            .padding(.bottom, MLXRSpacing.xl + 16)
        }
        .background(AdaptiveBackground())
    }

    private var header: some View {
        CompactPageHeader(
            title: "Home",
            subtitle: "Open a recent project, start something new, or jump to Models if the runtime still needs a starter row."
        ) {
            HStack(spacing: MLXRSpacing.xs) {
                StatusPill(
                    label: runtimeReady ? "Runtime ready" : "Connecting",
                    tint: runtimeReady ? MLXRColor.brandSecondary : MLXRColor.brandWarm
                )
                StatusPill(
                    label: hasModels ? "\(installedModelCount) installed" : "Models needed",
                    tint: hasModels ? MLXRColor.brandPrimary : MLXRColor.brandWarm
                )
                if activeJobCount > 0 {
                    StatusPill(
                        label: "\(activeJobCount) active",
                        tint: MLXRColor.brandPrimary
                    )
                }
                if queuedInstallCount > 0 {
                    StatusPill(
                        label: "\(queuedInstallCount) installing",
                        tint: MLXRColor.brandWarm
                    )
                }
                StatusPill(label: "\(totalCreations) outputs", tint: MLXRColor.brandSecondary)
            }
        }
    }

    private var quickActions: some View {
        DenseSectionSurface {
            VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                Text("Quick start")
                    .font(MLXRType.titleSmall)
                    .foregroundStyle(MLXRColor.textPrimary)

                HStack(spacing: MLXRSpacing.md) {
                    actionButton(
                        title: "Make image",
                        subtitle: "Generate or edit a still",
                        systemImage: "photo.fill",
                        tint: MLXRColor.brandPrimary,
                        action: onCreateImage
                    )
                    actionButton(
                        title: "Make video",
                        subtitle: "Text, image, audio, or retake",
                        systemImage: "film.fill",
                        tint: MLXRColor.brandSecondary,
                        action: onCreateVideo
                    )
                    actionButton(
                        title: "Manage models",
                        subtitle: "Install or remove rows",
                        systemImage: "square.stack.3d.up.fill",
                        tint: MLXRColor.brandWarm,
                        action: onOpenModels
                    )
                }
            }
        }
    }

    private var setupCard: some View {
        DenseSectionSurface {
            HStack {
                VStack(alignment: .leading, spacing: MLXRSpacing.xs) {
                    Text("Start with recommended models")
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                    Text("No models are installed in MLXR yet.")
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textPrimary)
                    Text("Open Models to queue the recommended image and video defaults.")
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textSecondary)
                }
                Spacer()
                Button("Open Models", action: onOpenModels)
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private var resumeCard: some View {
        DenseSectionSurface {
            VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                Text("Resume your workspace")
                    .font(MLXRType.titleSmall)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text("Your last draft is still waiting in Library, so you can keep iterating without rebuilding the setup.")
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)

                HStack(spacing: MLXRSpacing.md) {
                    Button("Open project", action: onResumeWorkspace)
                        .buttonStyle(.borderedProminent)
                    Button("Browse library", action: onOpenLibrary)
                        .buttonStyle(.bordered)
                }
            }
        }
    }

    private var firstCreationCard: some View {
        DenseSectionSurface {
            VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                Text("You’re ready to create")
                    .font(MLXRType.titleSmall)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text("Your starter models are installed. The next useful move is to make the first result and then build forward from it.")
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)

                HStack(spacing: MLXRSpacing.md) {
                    Button("Create an image", action: onCreateImage)
                        .buttonStyle(.borderedProminent)
                    Button("Create a video", action: onCreateVideo)
                        .buttonStyle(.bordered)
                }
            }
        }
    }

    private var continueCard: some View {
        DenseSectionSurface {
            VStack(alignment: .leading, spacing: MLXRSpacing.md) {
                Text("Continue from your library")
                    .font(MLXRType.titleSmall)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text("Everything you make or import becomes something you can use again.")
                    .font(MLXRType.bodyMedium)
                    .foregroundStyle(MLXRColor.textSecondary)
                ForEach(Array(recentAssets.prefix(5))) { asset in
                    Button {
                        onContinueAsset(asset)
                    } label: {
                        HStack(spacing: MLXRSpacing.md) {
                            Image(systemName: asset.isVideo ? "film" : asset.isAudio ? "waveform" : "photo")
                                .foregroundStyle(asset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary)
                                .frame(width: 18)
                            VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                                Text(asset.displayTitle)
                                    .font(.system(.headline, design: .rounded, weight: .semibold))
                                    .foregroundStyle(MLXRColor.textPrimary)
                                Text(asset.subtitle)
                                    .font(MLXRType.bodySmall)
                                    .foregroundStyle(MLXRColor.textSecondary)
                                    .lineLimit(2)
                            }
                            Spacer()
                            StatusPill(
                                label: asset.isImported ? "Imported" : asset.task?.title ?? "Generated",
                                tint: asset.isImported ? MLXRColor.brandWarm : MLXRColor.brandPrimary
                            )
                        }
                        .padding(.vertical, MLXRSpacing.xxs)
                    }
                    .buttonStyle(.plain)
                }

                Button("Open library", action: onOpenLibrary)
                    .buttonStyle(.bordered)
            }
        }
    }

    private func actionButton(
        title: String,
        subtitle: String,
        systemImage: String,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
                Image(systemName: systemImage)
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.system(.headline, design: .rounded, weight: .semibold))
                    .foregroundStyle(MLXRColor.textPrimary)
                Text(subtitle)
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
            }
            .frame(maxWidth: .infinity, minHeight: 94, alignment: .leading)
            .padding(MLXRSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(MLXRColor.surfaceCard)
                    .overlay(
                        RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                            .strokeBorder(tint.opacity(0.18), lineWidth: 0.5)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}
