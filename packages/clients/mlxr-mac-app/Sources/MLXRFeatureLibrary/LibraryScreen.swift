import AppKit
import AVKit
import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

public struct LibraryScreen: View {
    private let entries: [LibraryEntry]
    private let onMaterialize: @Sendable (OutputArtifactRecord) async -> URL?

    @State private var query = ""
    @State private var selectedEntryId: String?
    @State private var previewURL: URL?
    @State private var isMaterializing = false

    public init(
        entries: [LibraryEntry],
        onMaterialize: @escaping @Sendable (OutputArtifactRecord) async -> URL?
    ) {
        self.entries = entries
        self.onMaterialize = onMaterialize
    }

    public var body: some View {
        HSplitView {
            sidebarPane
                .frame(minWidth: 320)
                .padding(24)

            detailPane
                .frame(minWidth: 400)
                .padding(24)
        }
        .animation(MLXRAnimation.spring, value: selectedEntryId)
        .onAppear {
            if selectedEntryId == nil {
                selectedEntryId = filteredEntries.first?.id
            }
        }
        .onChange(of: selectedEntryId) { _, newValue in
            guard let newValue, let entry = entries.first(where: { $0.id == newValue }) else {
                previewURL = nil
                isMaterializing = false
                return
            }
            previewURL = nil
            isMaterializing = true
            Task {
                let url = await onMaterialize(entry.artifact)
                previewURL = url
                isMaterializing = false
            }
        }
    }

    // MARK: - Sidebar

    private var sidebarPane: some View {
        VStack(alignment: .leading, spacing: 16) {
            FeatureHeader(
                eyebrow: "Library",
                title: "Outputs",
                subtitle: "Browse every finished image and video artifact produced by the runtime."
            )

            TextField("Search prompt, model, or task", text: $query)
                .textFieldStyle(.roundedBorder)

            if filteredEntries.isEmpty {
                EmptyStateView(
                    title: "No outputs yet",
                    subtitle: "Run an image or video job and the finished artifacts will appear here automatically.",
                    systemImage: "shippingbox"
                )
            } else {
                List(filteredEntries, selection: $selectedEntryId) { entry in
                    entryRow(entry)
                        .tag(entry.id)
                }
                .listStyle(.inset)
            }
        }
    }

    private func entryRow(_ entry: LibraryEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(entry.title)
                .font(.system(.headline, design: .rounded, weight: .semibold))

            if !entry.prompt.isEmpty {
                Text(entry.prompt)
                    .font(.system(.subheadline, design: .rounded))
                    .lineLimit(2)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                StatusPill(
                    label: entry.task?.title ?? entry.job.request.task,
                    tint: MLXRTheme.accent
                )
                StatusPill(
                    label: entry.job.request.modelId,
                    tint: MLXRTheme.secondaryAccent
                )
            }
        }
    }

    // MARK: - Detail

    private var detailPane: some View {
        Group {
            if let selectedEntry {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        selectedDetailCard(selectedEntry)
                    }
                }
            } else {
                VStack {
                    Spacer()
                    EmptyStateView(
                        title: "Select an output",
                        subtitle: "Choose any image or video artifact to preview it and jump straight into Finder.",
                        systemImage: "rectangle.stack"
                    )
                    Spacer()
                }
            }
        }
    }

    private func selectedDetailCard(_ entry: LibraryEntry) -> some View {
        SectionCard(
            title: entry.title,
            subtitle: entry.prompt.isEmpty ? "Generated output" : entry.prompt
        ) {
            HStack(spacing: 10) {
                StatusPill(
                    label: entry.task?.title ?? entry.job.request.task,
                    tint: MLXRTheme.accent
                )
                StatusPill(
                    label: entry.job.request.modelId,
                    tint: MLXRTheme.secondaryAccent
                )
            }

            DetailRow(label: "Task", value: entry.job.request.task)
            DetailRow(label: "Model", value: entry.job.request.modelId)

            previewContent(for: entry)
                .frame(maxWidth: .infinity, minHeight: 420)

            if let previewURL {
                HStack(spacing: 12) {
                    Button("Reveal in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([previewURL])
                    }
                    .buttonStyle(.bordered)

                    Button("Open") {
                        NSWorkspace.shared.open(previewURL)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    @ViewBuilder
    private func previewContent(for entry: LibraryEntry) -> some View {
        if isMaterializing {
            VStack {
                Spacer()
                IndeterminateProgress(label: "Loading preview\u{2026}")
                Spacer()
            }
            .frame(maxWidth: .infinity)
        } else if let previewURL {
            PreviewPane(url: previewURL, mediaType: entry.artifact.mediaType)
        } else {
            EmptyStateView(
                title: "Preview not loaded",
                subtitle: "Select an artifact to materialize it into the local app cache.",
                systemImage: "eye"
            )
        }
    }

    // MARK: - Filtering

    private var filteredEntries: [LibraryEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return entries
        }
        let normalized = trimmed.lowercased()
        return entries.filter { entry in
            entry.prompt.lowercased().contains(normalized)
                || entry.job.request.modelId.lowercased().contains(normalized)
                || entry.job.request.task.lowercased().contains(normalized)
                || entry.title.lowercased().contains(normalized)
        }
    }

    private var selectedEntry: LibraryEntry? {
        guard let selectedEntryId else { return nil }
        return entries.first { $0.id == selectedEntryId }
    }
}

// MARK: - Preview Pane

private struct PreviewPane: View {
    let url: URL
    let mediaType: String?

    var body: some View {
        Group {
            if isImage, let nsImage = NSImage(contentsOf: url) {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            } else if isVideo {
                VideoPlayer(player: AVPlayer(url: url))
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            } else {
                EmptyStateView(
                    title: "Preview unavailable",
                    subtitle: "This artifact type can still be opened or revealed in Finder.",
                    systemImage: "doc"
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var isImage: Bool {
        mediaType?.hasPrefix("image/") == true
    }

    private var isVideo: Bool {
        mediaType?.hasPrefix("video/") == true || url.pathExtension.lowercased() == "mp4"
    }
}
