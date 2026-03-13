import MLXRAppDomain
import MLXRDesignSystem
import MLXRFeatureImages
import MLXRFeatureJobs
import MLXRFeatureLibrary
import MLXRFeatureModels
import MLXRFeatureSettings
import MLXRFeatureVideo
import SwiftUI

public struct MLXRMacAppRoot: View {
    @State private var appModel = MLXRAppModel()
    @State private var selectedSurface: ProductSurface? = .images

    public init() {}

    public var body: some View {
        ZStack {
            mainContent

            // Onboarding overlay
            if !appModel.hasCompletedOnboarding && !appModel.hasBootstrapped {
                onboardingOverlay
                    .transition(.opacity)
            }
        }
        .animation(MLXRAnimation.gentle, value: appModel.hasCompletedOnboarding)
        .task {
            await appModel.refresh()
        }
        .sheet(
            isPresented: Binding(
                get: { appModel.hasPendingModelSetup },
                set: { newValue in
                    if !newValue {
                        appModel.completeModelSetup()
                    }
                }
            )
        ) {
            FirstRunModelSetupView(
                models: appModel.catalog.recommendedAvailableItems,
                previews: appModel.modelPreviews,
                onLoadPreview: { modelId in
                    await appModel.loadPreviewIfNeeded(modelId: modelId)
                },
                onQueueInstall: { modelId in
                    await appModel.queueModelInstall(modelId: modelId)
                },
                onComplete: {
                    appModel.completeModelSetup()
                    selectedSurface = .models
                }
            )
        }
    }

    // MARK: - Main Content

    private var mainContent: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            ZStack(alignment: .top) {
                AdaptiveBackground()

                detailContent
                    .animation(MLXRAnimation.spring, value: selectedSurface)

                // Global error toast
                if let error = appModel.globalError {
                    VStack {
                        InlineErrorBanner(error) {
                            withAnimation(MLXRAnimation.snappy) {
                                appModel.dismissGlobalError()
                            }
                        }
                        .padding(.horizontal, 28)
                        .padding(.top, 8)
                        Spacer()
                    }
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .zIndex(100)
                }

                // Runtime process died warning
                if appModel.runtimeProcessDied {
                    VStack {
                        Spacer()
                        runtimeDeadBanner
                            .padding(20)
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                    .zIndex(99)
                }
            }
            .animation(MLXRAnimation.snappy, value: appModel.globalError != nil)
            .animation(MLXRAnimation.snappy, value: appModel.runtimeProcessDied)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        Task { await appModel.refresh() }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                            .symbolEffect(.rotate, isActive: appModel.isRefreshing)
                    }
                    .disabled(appModel.isRefreshing)
                    .keyboardShortcut("r", modifiers: .command)
                }
            }
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(ProductSurface.allCases, selection: $selectedSurface) { surface in
            SidebarItem(
                title: surface.title,
                systemImage: icon(for: surface),
                badge: badge(for: surface),
                isActive: selectedSurface == surface
            )
            .tag(surface)
        }
        .navigationTitle("MLXR")
    }

    // MARK: - Detail Content

    private var detailContent: some View {
        Group {
            switch selectedSurface ?? .images {
            case .images:
                ImagesScreen(
                    catalog: appModel.catalog,
                    isBusy: appModel.isSubmittingImage,
                    installOperations: appModel.installOperations,
                    error: appModel.imageError,
                    onDismissError: { appModel.dismissImageError() },
                    onInstall: { modelId in
                        await appModel.queueModelInstall(modelId: modelId)
                    },
                    onSubmit: { request in
                        await appModel.submitImage(request)
                    }
                )
            case .video:
                VideoScreen(
                    catalog: appModel.catalog,
                    isBusy: appModel.isSubmittingVideo,
                    installOperations: appModel.installOperations,
                    error: appModel.videoError,
                    onDismissError: { appModel.dismissVideoError() },
                    onInstall: { modelId in
                        await appModel.queueModelInstall(modelId: modelId)
                    },
                    onSubmit: { request in
                        await appModel.submitVideo(request)
                    }
                )
            case .models:
                ModelsScreen(
                    runtimeStatus: appModel.runtimeStatus,
                    catalog: appModel.catalog,
                    installOperations: appModel.installOperations,
                    previews: appModel.modelPreviews,
                    installedDetails: appModel.installedModelDetails,
                    error: appModel.modelsError,
                    onDismissError: { appModel.dismissModelsError() },
                    onLoadPreview: { modelId in
                        await appModel.loadPreviewIfNeeded(modelId: modelId)
                    },
                    onInstall: { modelId in
                        await appModel.queueModelInstall(modelId: modelId)
                    },
                    onCancelInstall: { operationId in
                        await appModel.cancelModelInstall(operationId: operationId)
                    },
                    onLoadDetails: { modelId in
                        await appModel.loadInstalledModelDetails(modelId: modelId)
                    },
                    onRemove: { modelId in
                        await appModel.removeInstalledModel(modelId: modelId)
                    }
                )
            case .library:
                LibraryScreen(
                    entries: appModel.libraryEntries,
                    onMaterialize: { artifact in
                        await appModel.cachedOutputURL(for: artifact)
                    }
                )
            case .jobs:
                JobsScreen(
                    jobs: appModel.jobs,
                    activePhases: appModel.activeJobPhases,
                    onCancel: { jobId in
                        await appModel.cancelJob(jobId: jobId)
                    },
                    onMaterialize: { artifact in
                        await appModel.cachedOutputURL(for: artifact)
                    }
                )
            case .settings:
                SettingsScreen(
                    runtimeStatus: appModel.runtimeStatus,
                    bootstrapError: appModel.bootstrapError,
                    error: appModel.settingsError,
                    onDismissError: { appModel.dismissSettingsError() },
                    promptHelperMode: $appModel.promptHelperMode,
                    onRefresh: {
                        await appModel.refresh()
                    },
                    onInspectImport: { draft in
                        try await appModel.inspectAdvancedImport(draft)
                    },
                    onRunImport: { draft in
                        try await appModel.runAdvancedImport(draft)
                    }
                )
            }
        }
    }

    // MARK: - Onboarding Overlay

    private var onboardingOverlay: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            VStack(spacing: 28) {
                Image(systemName: "sparkles")
                    .font(.system(size: 48, weight: .medium))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [MLXRTheme.accent, MLXRTheme.secondaryAccent],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .symbolEffect(.variableColor.iterative, options: .repeating.speed(0.5))

                VStack(spacing: 10) {
                    Text("Welcome to MLXR")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                    Text("Connecting to the local runtime...")
                        .font(.system(.body, design: .rounded))
                        .foregroundStyle(.secondary)
                }

                ProgressView()
                    .controlSize(.regular)
            }
            .padding(48)
            .background {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.regularMaterial)
                    .shadow(color: .black.opacity(0.15), radius: 30, y: 10)
            }
        }
    }

    // MARK: - Runtime Dead Banner

    private var runtimeDeadBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MLXRTheme.warmAccent)
                .font(.system(size: 16))
            VStack(alignment: .leading, spacing: 2) {
                Text("Runtime disconnected")
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                Text("The MLXR daemon has stopped. Refresh to reconnect.")
                    .font(.system(.caption, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Refresh") {
                Task { await appModel.refresh() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(MLXRTheme.warmAccent.opacity(0.3), lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.1), radius: 12, y: 4)
        }
    }

    // MARK: - Helpers

    private func icon(for surface: ProductSurface) -> String {
        switch surface {
        case .images:
            "photo.stack"
        case .video:
            "film.stack"
        case .models:
            "shippingbox"
        case .library:
            "square.stack.3d.up"
        case .jobs:
            "clock.arrow.trianglehead.counterclockwise.rotate.90"
        case .settings:
            "slider.horizontal.3"
        }
    }

    private func badge(for surface: ProductSurface) -> String? {
        switch surface {
        case .models:
            let count = appModel.activeInstallCount
            return count > 0 ? "\(count)" : nil
        case .jobs:
            let count = appModel.activeJobCount
            return count > 0 ? "\(count)" : nil
        default:
            return nil
        }
    }
}
