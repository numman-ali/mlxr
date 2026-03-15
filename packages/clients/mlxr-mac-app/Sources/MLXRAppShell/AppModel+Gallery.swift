import Foundation
import MLXRAppDomain

extension MLXRAppModel {
    public var defaultWorkspaceId: String {
        if workspaces.contains(where: { $0.id == "default-workspace" }) {
            return "default-workspace"
        }
        return workspaces.first?.id ?? activeWorkspaceId
    }

    public func resolvedWorkspaceId(for asset: LibraryAsset) -> String {
        if let workspaceId = asset.workspaceId {
            return workspaceId
        }
        if let runGroupId = asset.runGroupId,
           let runGroup = runGroupsByIdCache[runGroupId]
        {
            return runGroup.workspaceId
        }
        if let importedWorkspaceId = asset.importedAsset?.workspaceId {
            return importedWorkspaceId
        }
        return defaultWorkspaceId
    }

    public var hasRunnableCreationModels: Bool {
        catalog.installedItems.contains { !$0.tasks.isEmpty }
    }

    public var hasModelsReady: Bool {
        hasRunnableCreationModels
    }

    public var hasContent: Bool {
        hasContentCache
    }

    public var preferredLibraryWorkspaceId: String? {
        preferredLibraryWorkspaceIdCache
    }

    public var totalCreationCount: Int {
        totalCreationCountCache
    }

    public var activeJobs: [JobRecord] {
        jobs.filter { !$0.state.isTerminal }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public var latestActiveJob: JobRecord? {
        activeJobs.first
    }

    public var recentCompletedAssets: [LibraryAsset] {
        recentCompletedAssetsCache
    }

    public var latestSubmittedAsset: LibraryAsset? {
        latestSubmittedAssetCache
    }

    /// Progress estimate for a job based on its current phase
    public func estimatedProgress(for jobId: String) -> Double {
        guard let phase = activeJobPhases[jobId] else { return 0 }
        switch phase {
        case "accepted": return 0.05
        case "preparing": return 0.10
        case "loading_model": return 0.15
        case "running": return 0.50
        case "streaming_output": return 0.85
        case "finalizing": return 0.95
        default: return 0.30
        }
    }
}
