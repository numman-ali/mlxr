import MLXRActivityStrip
import MLXRAppDomain
import MLXRDesignSystem
import MLXRFeatureCreate
import MLXRFeatureGallery
import MLXRFeatureHome
import MLXRFeatureSettings
import MLXRFeatureToolkit
import SwiftUI

public struct MLXRMacAppRoot: View {
    @AppStorage("mlxr.mac-app.last-destination") private var lastDestinationRaw = Destination.home.rawValue
    @State private var appModel = MLXRAppModel()
    @State private var destination: Destination? = .home

    public init() {}

    public var body: some View {
        ZStack {
            if shouldShowBootstrapOverlay {
                onboardingOverlay
                    .zIndex(200)
                    .transition(.opacity)
            }

            HStack(spacing: 0) {
                NavRail(
                    selection: $destination,
                    items: { dest in
                        NavRailItemConfig(
                            icon: dest.icon,
                            label: dest.title,
                            badge: badge(for: dest)
                        )
                    },
                    footer: AnyView(
                        ActivityCenterButton(
                            jobs: appModel.jobs,
                            activePhases: appModel.activeJobPhases,
                            onCancelJob: { jobId in
                                await appModel.cancelJob(jobId: jobId)
                            },
                            onOpenLibrary: {
                                withAnimation(MLXRMotion.snappy) {
                                    destination = .library
                                }
                            },
                            presentation: .rail
                        )
                    )
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

                    shellUtilityBar
                        .padding(.horizontal, MLXRSpacing.lg)
                        .padding(.top, MLXRSpacing.md)
                        .padding(.bottom, MLXRSpacing.sm)

                    destinationContent
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if appModel.runtimeProcessDied {
                        runtimeDeadBanner
                            .padding(.horizontal, MLXRSpacing.lg)
                            .padding(.bottom, MLXRSpacing.md)
                    }
                }
            }
        }
        .task {
            await appModel.refresh()
            appModel.syncWorkspaceDefaultsForTask()
            chooseInitialDestinationIfNeeded()
        }
        .onChange(of: destination?.rawValue) { _, newValue in
            if let newValue {
                lastDestinationRaw = newValue
            }
        }
        .sheet(isPresented: Binding(
            get: { appModel.hasPendingModelSetup },
            set: { newValue in
                if !newValue {
                    appModel.completeModelSetup()
                }
            }
        )) {
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
            .interactiveDismissDisabled()
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
                    openStudio(task: .imageGenerate)
                },
                onCreateVideo: {
                    openStudio(task: .videoGenerate)
                },
                onOpenModels: {
                    withAnimation(MLXRMotion.snappy) {
                        destination = .models
                    }
                },
                onOpenLibrary: {
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                    }
                },
                onContinueAsset: { asset in
                    openStudio(
                        task: asset.isAudio ? .videoConditionAudio : asset.isVideo ? .videoRetake : .imageEdit,
                        focusedAssetId: asset.id,
                        referenceAssetIds: [asset.id]
                    )
                },
                onResumeWorkspace: {
                    withAnimation(MLXRMotion.snappy) {
                        destination = .studio
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
                isBusy: appModel.studioWorkspace.task.category == .image ? appModel.isSubmittingImage : appModel.isSubmittingVideo,
                error: appModel.studioWorkspace.task.category == .image ? appModel.imageError : appModel.videoError,
                onDismissError: {
                    if appModel.studioWorkspace.task.category == .image {
                        appModel.dismissImageError()
                    } else {
                        appModel.dismissVideoError()
                    }
                },
                onResetDraft: {
                    appModel.resetStudioDraft()
                },
                onSyncWorkspaceDefaults: {
                    appModel.syncWorkspaceDefaultsForTask()
                },
                onPrepareRunContext: { task, title, sourceAssetIds, variationCount in
                    let runGroup = appModel.createRunGroup(
                        task: task,
                        title: title.isEmpty ? task.title : title,
                        sourceAssetIds: sourceAssetIds,
                        variationCount: variationCount
                    )
                    return WorkflowContextMetadata(
                        workspaceId: appModel.activeWorkspaceId,
                        collectionId: nil,
                        runGroupId: runGroup.id,
                        sourceAssetIds: sourceAssetIds,
                        intentLabel: task.title,
                        presetId: appModel.studioWorkspace.aspectPreset.rawValue.lowercased()
                    )
                },
                onImportAssets: { urls in
                    await appModel.importExternalAssets(from: urls)
                },
                onResolveAssetURL: { asset in
                    await appModel.resolvedURL(for: asset)
                },
                onQueueInstall: { modelId in
                    await appModel.queueModelInstall(modelId: modelId)
                },
                onSubmitImage: { request in
                    await appModel.submitImage(request)
                },
                onSubmitVideo: { request in
                    await appModel.submitVideo(request)
                }
            )

        case .library:
            GalleryScreen(
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
                onOpenInStudio: { request in
                    openStudio(
                        task: request.task,
                        focusedAssetId: request.focusedAssetId,
                        referenceAssetIds: request.referenceAssetIds,
                        prompt: request.prompt
                    )
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

    private var shellUtilityBar: some View {
        HStack {
            StatusPill(
                label: appModel.runtimeStatus == nil ? "Connecting" : "Local runtime ready",
                tint: appModel.runtimeStatus == nil ? MLXRColor.brandWarm : MLXRColor.brandSecondary
            )
            Spacer()
        }
    }

    private var shouldShowBootstrapOverlay: Bool {
        !appModel.hasBootstrapped
            && appModel.bootstrapError == nil
            && appModel.globalError == nil
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
            let count = appModel.recentRunGroups.filter { $0.state == .running || $0.state == .queued }.count
            return count > 0 ? "\(count)" : nil
        default:
            return nil
        }
    }

    private func chooseInitialDestinationIfNeeded() {
        if appModel.hasWorkspaceDraft {
            destination = .studio
            return
        }
        if !appModel.hasModelsReady && !appModel.hasContent {
            destination = .home
            return
        }
        if let stored = Destination(rawValue: lastDestinationRaw) {
            destination = stored == .home ? .studio : stored
        } else {
            destination = .studio
        }
    }

    private func openStudio(
        task: ProductTask,
        focusedAssetId: String? = nil,
        referenceAssetIds: [String] = [],
        prompt: String? = nil
    ) {
        appModel.openStudio(
            task: task,
            focusedAssetId: focusedAssetId,
            referenceAssetIds: referenceAssetIds,
            prompt: prompt
        )
        withAnimation(MLXRMotion.snappy) {
            destination = .studio
        }
    }
}
