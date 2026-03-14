import Foundation
import MLXRAppDomain

extension MLXRAppModel {
    public var hasModelsReady: Bool {
        !catalog.installedItems.isEmpty
    }

    public var hasContent: Bool {
        !libraryAssets.isEmpty
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
