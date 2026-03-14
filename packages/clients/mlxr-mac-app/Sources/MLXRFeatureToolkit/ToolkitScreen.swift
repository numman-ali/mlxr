import MLXRAppDomain
import MLXRRuntimeBridge
import SwiftUI

@MainActor
public struct ToolkitScreen: View {
    let runtimeStatus: RuntimeStatusSnapshot?
    let catalog: CatalogSnapshot
    let installOperations: [ModelInstallOperationRecord]
    let modelPreviews: [String: SupportedModelPreview]
    let installedModelDetails: [String: InstalledModelDetails]
    let packs: [PackRecord]
    let error: String?
    let onDismissError: () -> Void
    let onQueueInstall: @Sendable (String) async -> Void
    let onCancelInstall: @Sendable (String) async -> Void
    let onRemoveModel: @Sendable (String) async -> Void
    let onLoadPreview: @Sendable (String) async -> Void
    let onLoadDetails: @Sendable (String) async -> Void

    public init(
        runtimeStatus: RuntimeStatusSnapshot?,
        catalog: CatalogSnapshot,
        installOperations: [ModelInstallOperationRecord],
        modelPreviews: [String: SupportedModelPreview],
        installedModelDetails: [String: InstalledModelDetails],
        packs: [PackRecord],
        error: String?,
        onDismissError: @escaping () -> Void,
        onQueueInstall: @escaping @Sendable (String) async -> Void,
        onCancelInstall: @escaping @Sendable (String) async -> Void,
        onRemoveModel: @escaping @Sendable (String) async -> Void,
        onLoadPreview: @escaping @Sendable (String) async -> Void,
        onLoadDetails: @escaping @Sendable (String) async -> Void
    ) {
        self.runtimeStatus = runtimeStatus
        self.catalog = catalog
        self.installOperations = installOperations
        self.modelPreviews = modelPreviews
        self.installedModelDetails = installedModelDetails
        self.packs = packs
        self.error = error
        self.onDismissError = onDismissError
        self.onQueueInstall = onQueueInstall
        self.onCancelInstall = onCancelInstall
        self.onRemoveModel = onRemoveModel
        self.onLoadPreview = onLoadPreview
        self.onLoadDetails = onLoadDetails
    }

    public var body: some View {
        ModelsWorkspaceView(
            runtimeStatus: runtimeStatus,
            catalog: catalog,
            installOperations: installOperations,
            previews: modelPreviews,
            installedDetails: installedModelDetails,
            packs: packs,
            error: error,
            onDismissError: onDismissError,
            onLoadPreview: onLoadPreview,
            onInstall: onQueueInstall,
            onCancelInstall: onCancelInstall,
            onLoadDetails: onLoadDetails,
            onRemove: onRemoveModel
        )
    }
}
