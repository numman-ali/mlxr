import MLXRAppDomain
import MLXRDesignSystem
import MLXRFeatureCreate
import MLXRFeatureGallery
import MLXRFeatureHome
import MLXRFeatureSettings
import MLXRFeatureToolkit
import MLXRActivityStrip
import SwiftUI

public struct MLXRMacAppRoot: View {
    private static let visibleDestinations: [Destination] = [.home, .library, .models, .settings]

    @AppStorage("mlxr.mac-app.last-destination") private var lastDestinationRaw = Destination.home.rawValue
    @State private var appModel = MLXRAppModel()
    @State private var destination: Destination? = .home
    @State private var isActivityPresented = false
    @State private var libraryFocusedWorkspaceId: String?
    @State private var libraryFocusedAssetId: String?

    public init() {}

    public var body: some View {
        ZStack {
            if shouldShowBootstrapOverlay {
                onboardingOverlay
                    .zIndex(200)
                    .transition(.opacity)
            }

            if isActivityPresented {
                activityOverlay
                    .zIndex(180)
                    .transition(.move(edge: .leading).combined(with: .opacity))
            }

            if appModel.hasPendingModelSetup {
                starterModelSetupOverlay
                    .zIndex(190)
                    .transition(.opacity)
            }

            HStack(spacing: 0) {
                NavRail(
                    selection: $destination,
                    destinations: Self.visibleDestinations,
                    items: { dest in
                        NavRailItemConfig(
                            icon: dest.icon,
                            label: dest.title,
                            badge: badge(for: dest)
                        )
                    },
                    footer: AnyView(railFooter)
                )

                VStack(spacing: 0) {
                    if let error = appModel.globalError {
                        InlineErrorBanner(error) {
                            appModel.dismissGlobalError()
                        }
                        .padding(.horizontal, MLXRSpacing.lg)
                            .padding(.top, MLXRSpacing.sm)
                            .padding(.bottom, MLXRSpacing.sm)
                    }

                    destinationContent
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if appModel.runtimeProcessDied {
                        runtimeDeadBanner
                            .padding(.horizontal, MLXRSpacing.lg)
                            .padding(.bottom, MLXRSpacing.md)
                    }
                }
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    if shouldShowGlobalComposer {
                        globalComposerInset
                    }
                }
            }
        }
        .task {
            await appModel.refresh()
            appModel.syncWorkspaceDefaultsForTask()
            appModel.scheduleStudioPlan()
            chooseInitialDestinationIfNeeded()
        }
        .onChange(of: destination?.rawValue) { _, newValue in
            if let newValue {
                lastDestinationRaw = newValue
            }
        }
        .onChange(of: appModel.studioWorkspace) { _, _ in
            appModel.scheduleStudioPlan()
        }
    }

    @ViewBuilder
    private var destinationContent: some View {
        switch destination ?? .home {
        case .home:
            HomeScreen(
                runtimeReady: appModel.hasBootstrapped,
                hasModels: appModel.hasModelsReady,
                hasContent: appModel.hasContent,
                installedModelCount: appModel.catalog.installedItems.count,
                activeJobCount: appModel.activeJobCount,
                queuedInstallCount: appModel.activeInstallCount,
                totalCreations: appModel.totalCreationCount,
                recentAssets: appModel.recentLibraryAssets,
                hasWorkspaceDraft: appModel.hasWorkspaceDraft,
                onCreateImage: {
                    appModel.selectComposerTask(.imageGenerate)
                },
                onCreateVideo: {
                    appModel.selectComposerTask(.videoGenerate)
                },
                onOpenModels: {
                    withAnimation(MLXRMotion.snappy) {
                        destination = .models
                    }
                },
                onOpenLibrary: {
                    appModel.showLibraryBrowser()
                    libraryFocusedWorkspaceId = nil
                    libraryFocusedAssetId = nil
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                    }
                },
                onContinueAsset: { asset in
                    libraryFocusedWorkspaceId = asset.workspaceId ?? appModel.activeWorkspaceId
                    libraryFocusedAssetId = asset.id
                    if let workspaceId = libraryFocusedWorkspaceId {
                        appModel.selectWorkspace(workspaceId)
                    }
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                    }
                },
                onResumeWorkspace: {
                    libraryFocusedWorkspaceId = appModel.activeWorkspaceId
                    libraryFocusedAssetId = nil
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                    }
                }
            )

        case .studio:
            StudioScreen(
                workspace: $appModel.studioWorkspace,
                catalog: appModel.catalog,
                libraryAssets: appModel.libraryAssets,
                installOperations: appModel.installOperations,
                jobs: appModel.jobs,
                activePhases: appModel.activeJobPhases,
                currentRunGroup: appModel.currentRunGroup,
                currentRunGroupAssets: appModel.currentRunGroupAssets,
                packs: appModel.packCatalog,
                planningResult: appModel.studioPlanResult,
                planningError: appModel.studioPlanError,
                isPlanning: appModel.isPlanningStudio,
                resolvedSettings: appModel.resolvedSettings(),
                error: appModel.studioWorkspace.task.category == .image ? appModel.imageError : appModel.videoError,
                onDismissError: {
                    if appModel.studioWorkspace.task.category == .image {
                        appModel.dismissImageError()
                    } else {
                        appModel.dismissVideoError()
                    }
                },
                onDismissPlanningError: {
                    appModel.dismissStudioPlanError()
                },
                onResetDraft: {
                    appModel.resetStudioDraft()
                },
                onSyncWorkspaceDefaults: {
                    appModel.syncWorkspaceDefaultsForTask()
                },
                onImportAssets: { urls in
                    await appModel.importExternalAssets(from: urls)
                },
                onResolveAssetURL: { asset in
                    await appModel.resolvedURL(for: asset)
                },
                onQueueInstall: { modelId in
                    await appModel.queueModelInstall(modelId: modelId)
                }
            )

        case .library:
            GalleryScreen(
                workspaces: appModel.workspaces,
                activeWorkspaceId: appModel.activeWorkspaceId,
                focusedWorkspaceId: libraryFocusedWorkspaceId ?? appModel.selectedLibraryWorkspaceId,
                focusedAssetId: libraryFocusedAssetId,
                assets: appModel.libraryAssets,
                runGroups: appModel.runGroups,
                collections: appModel.collections,
                onMaterialize: { asset in
                    await appModel.resolvedURL(for: asset)
                },
                onImportAssets: { urls in
                    await appModel.importExternalAssets(from: urls)
                },
                onRemoveImportedAsset: { assetId in
                    await appModel.removeImportedAsset(assetId: assetId)
                },
                onSeedComposer: { request in
                    seedComposer(request)
                },
                onSelectWorkspace: { workspaceId in
                    appModel.selectWorkspace(workspaceId)
                    libraryFocusedWorkspaceId = workspaceId
                },
                onCreateWorkspace: {
                    let workspace = appModel.createWorkspace()
                    libraryFocusedWorkspaceId = workspace.id
                    return workspace
                },
                onSetWorkspaceCover: { workspaceId, assetId in
                    appModel.setWorkspaceCoverAsset(workspaceId: workspaceId, assetId: assetId)
                },
                onToggleFavorite: { assetId in
                    appModel.toggleFavorite(assetId: assetId)
                },
                onToggleCollection: { assetId, collectionId in
                    appModel.toggleAsset(assetId, in: collectionId)
                },
                onCreateCollection: { title in
                    appModel.createCollection(named: title)
                }
            )

        case .models:
            ToolkitScreen(
                runtimeStatus: appModel.runtimeStatus,
                catalog: appModel.catalog,
                installOperations: appModel.installOperations,
                modelPreviews: appModel.modelPreviews,
                installedModelDetails: appModel.installedModelDetails,
                packs: appModel.packCatalog,
                error: appModel.modelsError,
                onDismissError: { appModel.dismissModelsError() },
                onQueueInstall: { modelId in
                    await appModel.queueModelInstall(modelId: modelId)
                },
                onCancelInstall: { operationId in
                    await appModel.cancelModelInstall(operationId: operationId)
                },
                onRemoveModel: { modelId in
                    await appModel.removeInstalledModel(modelId: modelId)
                },
                onLoadPreview: { modelId in
                    await appModel.loadPreviewIfNeeded(modelId: modelId)
                },
                onLoadDetails: { modelId in
                    await appModel.loadInstalledModelDetails(modelId: modelId)
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

    private var railFooter: some View {
        VStack(spacing: MLXRSpacing.sm) {
            Button {
                withAnimation(MLXRMotion.snappy) {
                    destination = .settings
                }
            } label: {
                VStack(spacing: 5) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(runtimeRailTint)
                            .frame(width: 8, height: 8)
                        Text(runtimeRailLabel)
                            .font(.system(size: 9, weight: .semibold, design: .rounded))
                            .foregroundStyle(MLXRColor.textSecondary)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .frame(width: 60)
                    .background(
                        RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                            .fill(MLXRColor.surfaceHover)
                            .overlay(
                                RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                                    .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                            )
                    )
                }
            }
            .buttonStyle(.plain)

            ActivityRailButton(
                runningCount: appModel.activityRunGroups.filter { $0.state == .running }.count + appModel.activeInstallCount,
                queuedCount: appModel.activityRunGroups.filter { $0.state == .queued }.count,
                hasFailures: appModel.activityRunGroups.contains { $0.state == .failed }
            ) {
                withAnimation(MLXRMotion.snappy) {
                    isActivityPresented.toggle()
                }
            }
        }
    }

    private var globalComposerInset: some View {
        GlobalComposerBar(
            workspace: $appModel.studioWorkspace,
            mode: appModel.composerMode,
            availableModes: appModel.composerAvailableModes,
            subworkflows: appModel.composerSubworkflowOptions(for: appModel.composerMode),
            qualityOptions: appModel.composerPresentation?.controls.qualityPresets ?? [],
            aspectOptions: appModel.composerPresentation?.controls.aspectPresets ?? [],
            durationOptions: appModel.composerPresentation?.controls.durationPresets ?? [],
            variationOptions: appModel.composerPresentation?.controls.variationCounts ?? [],
            selectedModelName: appModel.composerSelectedModelName,
            runtimeStatusLabel: appModel.composerRuntimeStatusLabel,
            isPlanning: appModel.isPlanningStudio,
            isBusy: appModel.isSubmittingImage || appModel.isSubmittingVideo,
            canSubmit: appModel.canSubmitCurrentWorkspace(),
            disabledReason: appModel.currentWorkspaceSubmitDisabledReason(),
            onSelectMode: { mode in
                appModel.selectComposerMode(mode)
            },
            onSelectTask: { task in
                appModel.selectComposerTask(task)
            },
            onSelectQuality: { value in
                appModel.selectComposerQuality(value)
            },
            onSelectAspect: { value in
                appModel.selectComposerAspect(value)
            },
            onSelectDuration: { value in
                appModel.selectComposerDuration(value)
            },
            onSelectVariation: { count in
                appModel.selectComposerVariationCount(count)
            },
            onSubmit: submitFromGlobalComposer
        )
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.bottom, MLXRSpacing.lg)
        .background(Color.clear)
    }

    private var activityOverlay: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 120)
            ActivityCenterSheet(
                runGroups: appModel.activityRunGroups,
                installOperations: appModel.installOperations,
                activePhases: appModel.activeJobPhases,
                onCancelRunGroup: { runGroupId in
                    await appModel.cancelRunGroup(runGroupId: runGroupId)
                },
                onDismissRunGroup: { runGroupId in
                    appModel.dismissActivityRunGroup(runGroupId)
                },
                onOpenLibrary: {
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                        isActivityPresented = false
                    }
                },
                onClose: {
                    withAnimation(MLXRMotion.snappy) {
                        isActivityPresented = false
                    }
                }
            )
            .padding(.leading, 90)
            .padding(.bottom, MLXRSpacing.lg)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
        .allowsHitTesting(true)
    }

    private var runtimeRailTint: Color {
        if appModel.runtimeProcessDied {
            return MLXRColor.brandDanger
        }
        return appModel.runtimeStatus == nil ? MLXRColor.brandWarm : MLXRColor.brandSecondary
    }

    private var runtimeRailLabel: String {
        if appModel.runtimeProcessDied {
            return "Down"
        }
        return appModel.runtimeStatus == nil ? "Boot" : "Ready"
    }

    private var shouldShowBootstrapOverlay: Bool {
        !appModel.hasBootstrapped
            && appModel.bootstrapError == nil
            && appModel.globalError == nil
    }

    private var shouldShowGlobalComposer: Bool {
        guard !shouldShowBootstrapOverlay else { return false }
        guard !appModel.hasPendingModelSetup else { return false }
        switch destination ?? .home {
        case .home, .library, .studio:
            return true
        case .models, .settings:
            return false
        }
    }

    private var onboardingOverlay: some View {
        ZStack {
            MLXRColor.canvasDeep
                .ignoresSafeArea()

            VStack(spacing: MLXRSpacing.xl) {
                ZStack {
                    Circle()
                        .fill(MLXRColor.brandGlow)
                        .frame(width: 92, height: 92)

                    Image(systemName: "sparkles")
                        .font(.system(size: 34, weight: .semibold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                }

                VStack(spacing: MLXRSpacing.xs) {
                    Text("MLXR")
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(MLXRColor.textPrimary)
                    Text("Connecting to the local runtime…")
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textSecondary)
                }

                ProgressView()
                    .tint(MLXRColor.brandPrimary)
            }
            .padding(MLXRSpacing.xxxl)
        }
    }

    private var starterModelSetupOverlay: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()

            StarterModelSetupView(
                catalog: appModel.catalog,
                previews: appModel.modelPreviews,
                onLoadPreview: { modelId in
                    await appModel.loadPreviewIfNeeded(modelId: modelId)
                },
                onQueueInstall: { modelId in
                    await appModel.queueModelInstall(modelId: modelId)
                },
                onComplete: {
                    appModel.completeModelSetup()
                },
                onSkip: {
                    appModel.completeModelSetup()
                }
            )
            .frame(maxWidth: 920, maxHeight: 760)
            .padding(MLXRSpacing.xl)
        }
    }

    private var runtimeDeadBanner: some View {
        HStack(spacing: MLXRSpacing.sm) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(MLXRColor.brandWarm)
                .font(.system(size: 16))

            VStack(alignment: .leading, spacing: 2) {
                Text("Runtime disconnected")
                    .font(MLXRType.bodySmall)
                    .fontWeight(.semibold)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text("Refresh in Settings if the daemon does not reconnect by itself.")
                    .font(MLXRType.captionLarge)
                    .foregroundStyle(MLXRColor.textSecondary)
            }

            Spacer()

            Button("Open Settings") {
                withAnimation(MLXRMotion.snappy) {
                    destination = .settings
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(MLXRSpacing.md)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                .fill(MLXRColor.surfaceCard)
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                        .strokeBorder(MLXRColor.brandWarm.opacity(0.25), lineWidth: 0.5)
                )
        }
    }

    private func badge(for dest: Destination) -> String? {
        switch dest {
        case .models:
            let count = appModel.activeInstallCount
            return count > 0 ? "\(count)" : nil
        case .library:
            let count = appModel.activityRunGroups.filter { $0.state == .running || $0.state == .queued }.count
            return count > 0 ? "\(count)" : nil
        default:
            return nil
        }
    }

    private func chooseInitialDestinationIfNeeded() {
        if appModel.hasPendingModelSetup {
            destination = .models
            return
        }
        if !appModel.hasModelsReady {
            destination = .home
            return
        }
        if appModel.hasContent {
            if let stored = Destination(rawValue: lastDestinationRaw), stored != .studio {
                destination = stored
            } else {
                destination = .library
            }
            libraryFocusedWorkspaceId = appModel.selectedLibraryWorkspaceId
        } else {
            destination = .home
        }
    }

    private func seedComposer(_ request: ComposerSeedRequest) {
        appModel.seedComposer(with: request)
        libraryFocusedWorkspaceId = request.workspaceId ?? appModel.activeWorkspaceId
        libraryFocusedAssetId = nil
    }

    private func submitFromGlobalComposer() {
        let targetWorkspaceId = appModel.studioWorkspace.workspaceId
        libraryFocusedWorkspaceId = targetWorkspaceId
        libraryFocusedAssetId = nil

        if destination != .library && destination != .studio {
            withAnimation(MLXRMotion.snappy) {
                destination = .library
            }
        }
        Task {
            await appModel.submitCurrentWorkspace()
        }
    }
}
