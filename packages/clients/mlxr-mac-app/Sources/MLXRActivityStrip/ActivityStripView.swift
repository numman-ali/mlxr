import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

@MainActor
public struct ActivityCenterButton: View {
    public enum Presentation {
        case toolbar
        case rail
    }

    let jobs: [JobRecord]
    let activePhases: [String: String]
    let onCancelJob: @Sendable (String) async -> Void
    let onOpenLibrary: () -> Void
    let presentation: Presentation

    @State private var isPresented = false

    public init(
        jobs: [JobRecord],
        activePhases: [String: String],
        onCancelJob: @escaping @Sendable (String) async -> Void,
        onOpenLibrary: @escaping () -> Void,
        presentation: Presentation = .toolbar
    ) {
        self.jobs = jobs
        self.activePhases = activePhases
        self.onCancelJob = onCancelJob
        self.onOpenLibrary = onOpenLibrary
        self.presentation = presentation
    }

    private var activeJobs: [JobRecord] {
        jobs.filter { !$0.state.isTerminal }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private var recentJobs: [JobRecord] {
        jobs.filter(\.state.isTerminal)
            .sorted { $0.updatedAt > $1.updatedAt }
            .prefix(8)
            .map { $0 }
    }

    public var body: some View {
        Button {
            isPresented.toggle()
        } label: {
            Group {
                switch presentation {
                case .toolbar:
                    HStack(spacing: MLXRSpacing.sm) {
                        Image(systemName: activeJobs.isEmpty ? "sparkles" : "waveform.path.ecg")
                            .foregroundStyle(activeJobs.isEmpty ? MLXRColor.textSecondary : MLXRColor.brandPrimary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(activeJobs.isEmpty ? "Activity" : "\(activeJobs.count) running")
                                .font(MLXRType.bodySmall)
                                .foregroundStyle(MLXRColor.textPrimary)
                            Text(activeJobs.first.flatMap { activePhases[$0.jobId] } ?? "Recent runs and results")
                                .font(MLXRType.captionLarge)
                                .foregroundStyle(MLXRColor.textTertiary)
                                .lineLimit(1)
                        }
                        if activeJobs.count > 0 {
                            StatusPill(label: "\(activeJobs.count)", tint: MLXRColor.brandPrimary)
                        }
                    }
                    .padding(.horizontal, MLXRSpacing.md)
                    .padding(.vertical, MLXRSpacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                            .fill(MLXRColor.surfaceCard)
                            .overlay(
                                RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                                    .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                            )
                    )
                case .rail:
                    VStack(spacing: 5) {
                        ZStack(alignment: .topTrailing) {
                            Image(systemName: activeJobs.isEmpty ? "sparkles" : "waveform.path.ecg")
                                .font(.system(size: 18, weight: activeJobs.isEmpty ? .regular : .semibold))
                                .foregroundStyle(activeJobs.isEmpty ? MLXRColor.textTertiary : MLXRColor.brandPrimary)
                                .frame(width: 36, height: 32)
                            if activeJobs.count > 0 {
                                Text("\(activeJobs.count)")
                                    .font(.system(size: 9, weight: .bold, design: .rounded))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 5)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(MLXRColor.brandWarm))
                                    .offset(x: 8, y: -6)
                            }
                        }
                        Text("Activity")
                            .font(.system(size: 10, weight: .medium, design: .rounded))
                            .foregroundStyle(MLXRColor.textTertiary)
                    }
                    .frame(width: 60, height: 56)
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
        }
        .buttonStyle(.plain)
        .popover(isPresented: $isPresented, attachmentAnchor: .point(.bottomTrailing)) {
            ActivityCenterPanel(
                activeJobs: activeJobs,
                recentJobs: recentJobs,
                activePhases: activePhases,
                onCancelJob: onCancelJob,
                onOpenLibrary: {
                    isPresented = false
                    onOpenLibrary()
                }
            )
            .frame(width: 380, height: 420)
            .padding(MLXRSpacing.lg)
            .background(MLXRColor.canvasRaised)
        }
    }
}

private struct ActivityCenterPanel: View {
    let activeJobs: [JobRecord]
    let recentJobs: [JobRecord]
    let activePhases: [String: String]
    let onCancelJob: @Sendable (String) async -> Void
    let onOpenLibrary: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Activity")
                        .font(MLXRType.titleMedium)
                        .foregroundStyle(MLXRColor.textPrimary)
                    Text("Running work stays here. Finished work lives in Library.")
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)
                }
                Spacer()
                Button("Open Library", action: onOpenLibrary)
                    .buttonStyle(.bordered)
            }

            if activeJobs.isEmpty {
                EmptyStateView(
                    title: "Nothing running",
                    subtitle: "Start from Studio and the current run will appear here while it works.",
                    systemImage: "sparkles"
                )
            } else {
                section(title: "Running now", jobs: activeJobs, canCancel: true)
            }

            if !recentJobs.isEmpty {
                section(title: "Recent", jobs: recentJobs, canCancel: false)
            }
        }
    }

    private func section(title: String, jobs: [JobRecord], canCancel: Bool) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(title.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            ForEach(jobs) { job in
                GlassCard {
                    HStack(alignment: .top, spacing: MLXRSpacing.sm) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(ProductTask.from(rawTask: job.request.task)?.title ?? job.request.task)
                                .font(MLXRType.bodyMedium)
                                .foregroundStyle(MLXRColor.textPrimary)
                            Text(job.request.inputs.string("prompt") ?? "")
                                .font(MLXRType.captionLarge)
                                .foregroundStyle(MLXRColor.textSecondary)
                                .lineLimit(2)
                            Text(activePhases[job.jobId] ?? job.state.displayName)
                                .font(MLXRType.captionSmall)
                                .foregroundStyle(MLXRColor.textTertiary)
                        }
                        Spacer()
                        if canCancel {
                            Button("Cancel") {
                                Task { await onCancelJob(job.jobId) }
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        } else {
                            StatusPill(label: job.state.displayName, tint: tint(for: job.state))
                        }
                    }
                }
            }
        }
    }

    private func tint(for state: JobState) -> Color {
        switch state {
        case .completed:
            MLXRColor.brandSecondary
        case .failed:
            MLXRColor.brandDanger
        case .cancelled:
            MLXRColor.brandWarm
        default:
            MLXRColor.brandPrimary
        }
    }
}

private extension JobState {
    var displayName: String {
        switch self {
        case .accepted: "Accepted"
        case .preparing: "Preparing"
        case .loadingModel: "Loading model"
        case .running: "Running"
        case .streamingOutput: "Streaming output"
        case .finalizing: "Finalizing"
        case .completed: "Completed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }
}
