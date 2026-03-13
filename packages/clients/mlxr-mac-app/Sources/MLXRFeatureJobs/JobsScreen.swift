import AppKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

public struct JobsScreen: View {
    private let jobs: [JobRecord]
    private let activePhases: [String: String]
    private let onCancel: @Sendable (String) async -> Void
    private let onMaterialize: @Sendable (OutputArtifactRecord) async -> URL?

    public init(
        jobs: [JobRecord],
        activePhases: [String: String],
        onCancel: @escaping @Sendable (String) async -> Void,
        onMaterialize: @escaping @Sendable (OutputArtifactRecord) async -> URL?
    ) {
        self.jobs = jobs
        self.activePhases = activePhases
        self.onCancel = onCancel
        self.onMaterialize = onMaterialize
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                FeatureHeader(
                    eyebrow: "Jobs",
                    title: "Activity",
                    subtitle: "The app listens to the same runtime job stream as the CLI path, so this surface is honest about progress, failure, and outputs."
                )

                if jobs.isEmpty {
                    EmptyStateView(
                        title: "No jobs yet",
                        subtitle: "Once you run something, active work and recent completions will show up here.",
                        systemImage: "hourglass"
                    )
                } else {
                    ForEach(jobs) { job in
                        SectionCard(
                            title: job.request.task,
                            subtitle: job.request.inputs.string("prompt") ?? job.request.modelId
                        ) {
                            HStack(spacing: 10) {
                                StatusPill(label: job.state.rawValue, tint: tint(for: job.state))
                                StatusPill(label: job.request.modelId, tint: MLXRTheme.secondaryAccent)
                                MetricBadge(label: "Artifacts", value: "\(job.artifacts.count)")
                            }

                            if let phase = activePhases[job.jobId] {
                                HStack(spacing: 8) {
                                    ProgressView()
                                        .controlSize(.small)
                                    Text(phase)
                                        .font(.system(.subheadline, design: .rounded, weight: .medium))
                                        .foregroundStyle(.secondary)
                                }
                                .transition(.opacity.combined(with: .blurReplace))
                            }

                            if let error = job.error, !error.isEmpty {
                                InlineErrorBanner(error)
                            }

                            if !job.artifacts.isEmpty {
                                VStack(alignment: .leading, spacing: 8) {
                                    Text("Artifacts")
                                        .font(.system(.headline, design: .rounded, weight: .semibold))
                                    ForEach(job.artifacts) { artifact in
                                        ArtifactRow(artifact: artifact, onMaterialize: onMaterialize)
                                    }
                                }
                            }

                            if !job.state.isTerminal {
                                Button("Cancel job") {
                                    Task { await onCancel(job.jobId) }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                        .transition(.asymmetric(
                            insertion: .move(edge: .top).combined(with: .opacity),
                            removal: .opacity
                        ))
                    }
                }
            }
            .padding(28)
            .animation(MLXRAnimation.spring, value: jobs.map(\.state))
            .animation(MLXRAnimation.spring, value: activePhases)
        }
    }

    private func tint(for state: JobState) -> Color {
        switch state {
        case .completed:
            MLXRTheme.secondaryAccent
        case .failed, .cancelled:
            MLXRTheme.destructive
        case .running, .loadingModel, .preparing, .streamingOutput, .finalizing:
            MLXRTheme.warmAccent
        case .accepted:
            MLXRTheme.accent
        }
    }
}

private struct ArtifactRow: View {
    let artifact: OutputArtifactRecord
    let onMaterialize: @Sendable (OutputArtifactRecord) async -> URL?

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(artifact.filename ?? artifact.artifactId)
                Text(artifact.mediaType ?? artifact.artifactFormat)
                    .font(.system(.caption, design: .rounded))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Reveal in Finder") {
                Task {
                    guard let url = await onMaterialize(artifact) else {
                        return
                    }
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            }
            .buttonStyle(.borderless)
        }
    }
}
