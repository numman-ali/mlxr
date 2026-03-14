import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

public struct ActivityRailButton: View {
    let runningCount: Int
    let queuedCount: Int
    let hasFailures: Bool
    let action: () -> Void

    public init(
        runningCount: Int,
        queuedCount: Int,
        hasFailures: Bool,
        action: @escaping () -> Void
    ) {
        self.runningCount = runningCount
        self.queuedCount = queuedCount
        self.hasFailures = hasFailures
        self.action = action
    }

    private var badgeCount: Int {
        runningCount + queuedCount
    }

    private var iconName: String {
        if hasFailures {
            return "exclamationmark.bubble.fill"
        }
        if runningCount > 0 {
            return "waveform.path.ecg"
        }
        if queuedCount > 0 {
            return "clock.badge"
        }
        return "sparkles"
    }

    private var iconTint: Color {
        if hasFailures {
            return MLXRColor.brandDanger
        }
        if runningCount > 0 {
            return MLXRColor.brandPrimary
        }
        if queuedCount > 0 {
            return MLXRColor.brandWarm
        }
        return MLXRColor.textTertiary
    }

    public var body: some View {
        Button(action: action) {
            VStack(spacing: 5) {
                ZStack(alignment: .topTrailing) {
                    Image(systemName: iconName)
                        .font(.system(size: 18, weight: runningCount > 0 ? .semibold : .regular))
                        .foregroundStyle(iconTint)
                        .frame(width: 36, height: 32)

                    if badgeCount > 0 {
                        Text("\(badgeCount)")
                            .font(.system(size: 9, weight: .bold, design: .rounded))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(hasFailures ? MLXRColor.brandDanger : MLXRColor.brandWarm))
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
        .buttonStyle(.plain)
    }
}

public struct ActivityCenterSheet: View {
    let runGroups: [RunGroupRecord]
    let installOperations: [ModelInstallOperationRecord]
    let activePhases: [String: String]
    let onCancelRunGroup: @Sendable (String) async -> Void
    let onDismissRunGroup: (String) -> Void
    let onOpenLibrary: () -> Void
    let onClose: () -> Void

    public init(
        runGroups: [RunGroupRecord],
        installOperations: [ModelInstallOperationRecord],
        activePhases: [String: String],
        onCancelRunGroup: @escaping @Sendable (String) async -> Void,
        onDismissRunGroup: @escaping (String) -> Void,
        onOpenLibrary: @escaping () -> Void,
        onClose: @escaping () -> Void
    ) {
        self.runGroups = runGroups
        self.installOperations = installOperations
        self.activePhases = activePhases
        self.onCancelRunGroup = onCancelRunGroup
        self.onDismissRunGroup = onDismissRunGroup
        self.onOpenLibrary = onOpenLibrary
        self.onClose = onClose
    }

    private var activeInstallOperations: [ModelInstallOperationRecord] {
        installOperations.filter { !$0.phase.isTerminal }
    }

    private var queuedGroups: [RunGroupRecord] {
        runGroups.filter { $0.state == .queued }
    }

    private var runningGroups: [RunGroupRecord] {
        runGroups.filter { $0.state == .running }
    }

    private var failedGroups: [RunGroupRecord] {
        runGroups.filter { $0.state == .failed }.prefix(8).map { $0 }
    }

    private var recentCompletedGroups: [RunGroupRecord] {
        runGroups.filter { $0.state == .completed || $0.state == .cancelled }
            .prefix(10)
            .map { $0 }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
            header

            if runningGroups.isEmpty
                && queuedGroups.isEmpty
                && failedGroups.isEmpty
                && recentCompletedGroups.isEmpty
                && activeInstallOperations.isEmpty
            {
                EmptyStateView(
                    title: "No recent activity",
                    subtitle: "Runs you start from the composer appear here while they work, then move into the Library once they finish.",
                    systemImage: "sparkles"
                )
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                        if !activeInstallOperations.isEmpty {
                            installSection
                        }
                        if !runningGroups.isEmpty {
                            section(title: "Running now", groups: runningGroups)
                        }
                        if !queuedGroups.isEmpty {
                            section(title: "Queued", groups: queuedGroups)
                        }
                        if !failedGroups.isEmpty {
                            section(title: "Needs attention", groups: failedGroups)
                        }
                        if !recentCompletedGroups.isEmpty {
                            section(title: "Recent", groups: recentCompletedGroups)
                        }
                    }
                    .padding(.bottom, MLXRSpacing.xl)
                }
            }
        }
        .padding(MLXRSpacing.lg)
        .frame(width: 440)
        .frame(maxHeight: 620)
        .background {
            RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                .fill(MLXRColor.canvasRaised)
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
                .shadow(color: .black.opacity(0.18), radius: 24, y: 8)
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Activity")
                    .font(MLXRType.titleLarge)
                    .foregroundStyle(MLXRColor.textPrimary)
                Text("Accepted runs and installs. Finished work lands in Library.")
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
            }

            Spacer()

            HStack(spacing: MLXRSpacing.sm) {
                Button("Open Library") {
                    onClose()
                    onOpenLibrary()
                }
                .buttonStyle(.bordered)

                Button("Close") {
                    onClose()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var installSection: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text("INSTALLS")
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)

            LazyVStack(spacing: MLXRSpacing.sm) {
                ForEach(activeInstallOperations) { operation in
                    ActivityInstallCard(operation: operation)
                }
            }
        }
    }

    private func section(title: String, groups: [RunGroupRecord]) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(title.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)

            LazyVStack(spacing: MLXRSpacing.sm) {
                ForEach(groups) { group in
                    ActivityRunGroupCard(
                        group: group,
                        phase: phaseSummary(for: group),
                        outputCount: outputCount(for: group),
                        canCancel: group.state == .queued || group.state == .running,
                        canDismiss: group.state == .failed || group.state == .completed || group.state == .cancelled,
                        onCancel: {
                            await onCancelRunGroup(group.id)
                        },
                        onDismiss: {
                            onDismissRunGroup(group.id)
                        }
                    )
                }
            }
        }
    }

    private func phaseSummary(for group: RunGroupRecord) -> String {
        for jobId in group.jobIds {
            if let phase = activePhases[jobId], !phase.isEmpty {
                return phase
            }
        }
        return group.state.label
    }

    private func outputCount(for group: RunGroupRecord) -> Int {
        max(group.assetIds.count, group.variationCount)
    }
}

private struct ActivityRunGroupCard: View {
    let group: RunGroupRecord
    let phase: String
    let outputCount: Int
    let canCancel: Bool
    let canDismiss: Bool
    let onCancel: @Sendable () async -> Void
    let onDismiss: () -> Void

    var body: some View {
        GlassCard {
            HStack(alignment: .top, spacing: MLXRSpacing.md) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: MLXRSpacing.xs) {
                        Text(group.task.title)
                            .font(MLXRType.bodyMedium)
                            .foregroundStyle(MLXRColor.textPrimary)

                        StatusPill(label: group.state.label, tint: tint(for: group.state))
                    }

                    Text(group.title)
                        .font(MLXRType.titleSmall)
                        .foregroundStyle(MLXRColor.textPrimary)
                        .lineLimit(2)

                    Text(phase)
                        .font(MLXRType.bodySmall)
                        .foregroundStyle(MLXRColor.textSecondary)

                    Text(metadataLine)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textTertiary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: MLXRSpacing.xs) {
                    if canCancel {
                        Button("Cancel") {
                            Task { await onCancel() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    if canDismiss {
                        Button("Dismiss") {
                            onDismiss()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
            }
        }
    }

    private var metadataLine: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        let updated = formatter.localizedString(for: group.updatedAt, relativeTo: .now)
        return "\(outputCount) output\(outputCount == 1 ? "" : "s") • \(updated)"
    }

    private func tint(for state: RunGroupState) -> Color {
        switch state {
        case .queued:
            MLXRColor.brandWarm
        case .running:
            MLXRColor.brandPrimary
        case .completed:
            MLXRColor.brandSecondary
        case .failed:
            MLXRColor.brandDanger
        case .cancelled:
            MLXRColor.textTertiary
        }
    }
}

private struct ActivityInstallCard: View {
    let operation: ModelInstallOperationRecord

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: MLXRSpacing.xs) {
                    Text(operation.supportedModel.displayName)
                        .font(MLXRType.bodyMedium)
                        .foregroundStyle(MLXRColor.textPrimary)
                    Spacer()
                    StatusPill(label: phaseLabel, tint: tint)
                }

                Text(operation.supportedModel.sourceSummary)
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)

                if let authMessage = operation.preview?.authMessage, !authMessage.isEmpty {
                    Text(authMessage)
                        .font(MLXRType.captionLarge)
                        .foregroundStyle(MLXRColor.textTertiary)
                }
            }
        }
    }

    private var phaseLabel: String {
        switch operation.phase {
        case .queued: "Queued"
        case .resolving: "Resolving"
        case .authRequired: "Requires login"
        case .downloading: "Downloading"
        case .converting: "Converting"
        case .registering: "Registering"
        case .completed: "Installed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }

    private var tint: Color {
        switch operation.phase {
        case .queued:
            MLXRColor.brandWarm
        case .resolving, .downloading, .converting, .registering:
            MLXRColor.brandPrimary
        case .authRequired:
            MLXRColor.brandWarm
        case .completed:
            MLXRColor.brandSecondary
        case .failed:
            MLXRColor.brandDanger
        case .cancelled:
            MLXRColor.textTertiary
        }
    }
}

private extension RunGroupState {
    var label: String {
        switch self {
        case .queued: "Queued"
        case .running: "Running"
        case .completed: "Completed"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        }
    }
}
