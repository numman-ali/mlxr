import MLXRAppDomain
import MLXRDesignSystem
import MLXRFeatureCreate
import MLXRFeatureGallery
import MLXRFeatureHome
import MLXRFeatureSettings
import MLXRFeatureToolkit
import MLXRActivityStrip
import SwiftUI

private enum ComposerChromeState: String {
    case hidden
    case collapsed
    case expanded
}

public struct MLXRMacAppRoot: View {
    private static let visibleDestinations: [Destination] = [.home, .library, .models, .settings]
    private static let composerCourtesyInset: CGFloat = 112

    @AppStorage("mlxr.mac-app.last-destination") private var lastDestinationRaw = Destination.home.rawValue
    @AppStorage("mlxr.mac-app.composer-chrome-state") private var composerChromeStateRaw = ComposerChromeState.collapsed.rawValue
    @State private var appModel = MLXRAppModel()
    @State private var destination: Destination? = .home
    @State private var isActivityPresented = false
    @State private var libraryFocusedWorkspaceId: String?
    @State private var libraryFocusedAssetId: String?
    @State private var isLibraryViewerPresented = false

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
                        .environment(\.mlxrBottomOverlayInset, composerBottomInset)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if appModel.runtimeProcessDied {
                        runtimeDeadBanner
                            .padding(.horizontal, MLXRSpacing.lg)
                            .padding(.bottom, MLXRSpacing.md)
                    }
                }
                .overlay(alignment: .bottom) {
                    if shouldShowGlobalComposer {
                        composerOverlay
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
            }
        }
        .task {
            await appModel.refresh()
            appModel.syncWorkspaceDefaultsForTask()
            appModel.scheduleCreationPlan()
            chooseInitialDestinationIfNeeded()
        }
        .onChange(of: destination?.rawValue) { _, newValue in
            if let newValue {
                lastDestinationRaw = newValue
            }
        }
        .onChange(of: appModel.creationDraft) { _, _ in
            appModel.scheduleCreationPlan()
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
                    appModel.clearProjectScopedDraftContext()
                    composerChromeState = .expanded
                    appModel.selectComposerTask(.imageGenerate)
                },
                onCreateVideo: {
                    appModel.clearProjectScopedDraftContext()
                    composerChromeState = .expanded
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
                    libraryFocusedWorkspaceId = appModel.resolvedWorkspaceId(for: asset)
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
                    composerChromeState = .expanded
                    withAnimation(MLXRMotion.snappy) {
                        destination = .library
                    }
                }
            )

        case .library:
            GalleryScreen(
                workspaces: appModel.workspaces,
                defaultWorkspaceId: appModel.defaultWorkspaceId,
                focusedWorkspaceId: libraryFocusedWorkspaceId ?? appModel.selectedLibraryWorkspaceId,
                focusedAssetId: libraryFocusedAssetId,
                presentationDataRevision: appModel.libraryPresentationRevision,
                assets: appModel.libraryAssets,
                runGroups: appModel.runGroups,
                collections: appModel.collections,
                onMaterialize: { asset in
                    await appModel.resolvedURL(for: asset)
                },
                onImportAssets: { urls, workspaceId in
                    await appModel.importExternalAssets(from: urls, workspaceId: workspaceId)
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
                onShowProjectBrowser: {
                    appModel.showLibraryBrowser()
                    libraryFocusedWorkspaceId = nil
                    libraryFocusedAssetId = nil
                },
                onCreateWorkspace: {
                    let workspace = appModel.createFreshWorkspace()
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
                },
                onViewerPresentationChange: { isPresented in
                    isLibraryViewerPresented = isPresented
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

    private var composerChromeState: ComposerChromeState {
        get { ComposerChromeState(rawValue: composerChromeStateRaw) ?? .collapsed }
        nonmutating set { composerChromeStateRaw = newValue.rawValue }
    }

    private var composerBottomInset: CGFloat {
        guard shouldShowGlobalComposer else { return 0 }
        switch composerChromeState {
        case .hidden:
            return 0
        case .collapsed:
            return 0
        case .expanded:
            // Keep a stable courtesy inset so the shell reads as an overlay,
            // not a footer that continuously resizes the content behind it.
            return Self.composerCourtesyInset
        }
    }

    private var composerOverlay: some View {
        Group {
            switch composerChromeState {
            case .hidden:
                composerRevealPill
            case .collapsed, .expanded:
                globalComposerInset
            }
        }
    }

    private var globalComposerInset: some View {
        GlobalComposerBar(
            workspace: $appModel.creationDraft,
            isCollapsed: composerChromeState == .collapsed,
            mode: appModel.composerMode,
            availableModes: appModel.composerAvailableModes,
            subworkflows: appModel.composerSubworkflowOptions(for: appModel.composerMode),
            qualityOptions: appModel.composerPresentation?.controls.qualityPresets ?? [],
            aspectOptions: appModel.composerPresentation?.controls.aspectPresets ?? [],
            durationOptions: appModel.composerPresentation?.controls.durationPresets ?? [],
            variationOptions: appModel.composerPresentation?.controls.variationCounts ?? [],
            selectedModelName: appModel.composerSelectedModelName,
            selectedAssetLabel: appModel.composerFocusedAssetLabel,
            referenceSummaryLabel: appModel.composerReferenceSummaryLabel,
            runtimeStatusLabel: appModel.composerRuntimeStatusLabel,
            isPlanning: appModel.isPlanningCreation,
            isBusy: appModel.isSubmittingImage || appModel.isSubmittingVideo,
            canSubmit: appModel.canSubmitCurrentWorkspace(),
            disabledReason: appModel.currentWorkspaceSubmitDisabledReason(),
            onCollapse: {
                withAnimation(MLXRMotion.snappy) {
                    composerChromeState = .collapsed
                }
            },
            onExpand: {
                withAnimation(MLXRMotion.snappy) {
                    composerChromeState = .expanded
                }
            },
            onHide: {
                withAnimation(MLXRMotion.snappy) {
                    composerChromeState = .hidden
                }
            },
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
        .frame(maxWidth: composerChromeState == .collapsed ? 460 : 1_020)
        .frame(maxWidth: .infinity, alignment: composerChromeState == .collapsed ? .trailing : .center)
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.bottom, MLXRSpacing.lg)
        .padding(.top, MLXRSpacing.md)
        .background(Color.clear)
    }

    private var composerRevealPill: some View {
        Button {
            withAnimation(MLXRMotion.snappy) {
                composerChromeState = .expanded
            }
        } label: {
            HStack(spacing: MLXRSpacing.sm) {
                Image(systemName: appModel.composerMode == .image ? "photo.on.rectangle" : "film.stack")
                    .font(.system(size: 12, weight: .semibold))
                Text("Open composer")
                    .font(MLXRType.bodySmall)
                    .fontWeight(.semibold)
                StatusPill(
                    label: appModel.composerRuntimeStatusLabel,
                    tint: appModel.canSubmitCurrentWorkspace() ? MLXRColor.brandPrimary : MLXRColor.brandWarm
                )
            }
            .padding(.horizontal, MLXRSpacing.md)
            .padding(.vertical, MLXRSpacing.sm)
        }
        .buttonStyle(.plain)
        .background(
            Capsule(style: .continuous)
                .fill(MLXRColor.canvasRaised)
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.75)
                )
                .shadow(color: .black.opacity(0.14), radius: 12, y: 6)
        )
        .padding(.horizontal, MLXRSpacing.xl)
        .padding(.bottom, MLXRSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .trailing)
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
        guard appModel.hasRunnableCreationModels else { return false }
        guard !isLibraryViewerPresented else { return false }
        switch destination ?? .home {
        case .home, .library:
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
        if appModel.hasContent {
            destination = .library
            libraryFocusedWorkspaceId = appModel.selectedLibraryWorkspaceId ?? appModel.preferredLibraryWorkspaceId
        } else {
            destination = .home
        }
    }

    private func seedComposer(_ request: ComposerSeedRequest) {
        appModel.seedComposer(with: request)
        if let workspaceId = request.workspaceId {
            libraryFocusedWorkspaceId = workspaceId
        } else if let focusedAssetId = request.focusedAssetId,
                  let asset = appModel.libraryAssets.first(where: { $0.id == focusedAssetId })
        {
            libraryFocusedWorkspaceId = appModel.resolvedWorkspaceId(for: asset)
        } else {
            libraryFocusedWorkspaceId = appModel.activeWorkspaceId
        }
        libraryFocusedAssetId = nil
        composerChromeState = .expanded
    }

    private func submitFromGlobalComposer() {
        Task {
            let targetWorkspaceId = freshSubmissionWorkspaceId
            guard let acceptedWorkspaceId = await appModel.submitCurrentWorkspace(targetWorkspaceId: targetWorkspaceId) else {
                return
            }
            libraryFocusedWorkspaceId = acceptedWorkspaceId
            libraryFocusedAssetId = nil
            appModel.selectWorkspace(acceptedWorkspaceId, clearingDraftContext: false)

            if destination == .home {
                withAnimation(MLXRMotion.snappy) {
                    destination = .library
                }
            }
        }
    }

    private var freshSubmissionWorkspaceId: String? {
        if destination == .home {
            return WorkspaceRecord(title: "New Project").id
        }
        if isTopLevelLibraryBrowser {
            return WorkspaceRecord(title: "New Project").id
        }
        return nil
    }

    private var isTopLevelLibraryBrowser: Bool {
        destination == .library
            && libraryFocusedWorkspaceId == nil
            && appModel.selectedLibraryWorkspaceId == nil
    }
}
