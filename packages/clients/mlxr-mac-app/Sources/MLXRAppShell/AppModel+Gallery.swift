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
           let runGroup = runGroups.first(where: { $0.id == runGroupId })
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
        !libraryAssets.isEmpty
    }

    public var preferredLibraryWorkspaceId: String? {
        if let selectedLibraryWorkspaceId,
           libraryAssets.contains(where: { resolvedWorkspaceId(for: $0) == selectedLibraryWorkspaceId })
        {
            return selectedLibraryWorkspaceId
        }
        if libraryAssets.contains(where: { resolvedWorkspaceId(for: $0) == activeWorkspaceId }) {
            return activeWorkspaceId
        }
        if libraryAssets.contains(where: { resolvedWorkspaceId(for: $0) == defaultWorkspaceId }) {
            return defaultWorkspaceId
        }
        return libraryAssets.first.map(resolvedWorkspaceId(for:))
    }

    public var totalCreationCount: Int {
        jobs.filter { $0.state == .completed }.count
    }

    public var activeJobs: [JobRecord] {
        jobs.filter { !$0.state.isTerminal }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public var latestActiveJob: JobRecord? {
        activeJobs.first
    }

    public var recentCompletedAssets: [LibraryAsset] {
        libraryAssets
            .filter(\.isGenerated)
            .sorted { $0.createdAt > $1.createdAt }
    }

    public var latestSubmittedAsset: LibraryAsset? {
        guard let lastSubmittedJobId else { return nil }
        return libraryAssets.first { $0.jobId == lastSubmittedJobId }
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
