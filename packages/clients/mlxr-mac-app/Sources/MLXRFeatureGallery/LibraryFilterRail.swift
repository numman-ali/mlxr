import MLXRAppDomain
import MLXRDesignSystem
import SwiftUI

struct LibraryFilterRail: View {
    @Binding var selectedFilter: LibraryAssetFilter
    @Binding var selectedModelId: String
    @Binding var selectedTaskRaw: String
    @Binding var selectedSort: AssetSortMode
    @Binding var favoritesOnly: Bool
    @Binding var selectedCollectionId: String?
    @Binding var newCollectionName: String

    let presentation: LibraryPresentationModel
    let onCreateCollection: (String) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: MLXRSpacing.lg) {
                FeatureHeader(
                    eyebrow: "Library",
                    title: "Everything you made or imported",
                    subtitle: "Browse reusable assets, keep lightweight collections, and send anything straight back into Studio."
                )

                GlassCard(title: "Filter", subtitle: "Narrow the library to the media you need right now.") {
                    filterSection("Media") {
                        ForEach(LibraryAssetFilter.allCases) { filter in
                            filterButton(
                                title: filter.rawValue,
                                isSelected: selectedFilter == filter,
                                count: presentation.count(for: filter)
                            ) {
                                selectedFilter = filter
                            }
                        }
                    }

                    Toggle("Favorites only", isOn: $favoritesOnly)
                        .toggleStyle(.switch)

                    Picker("Model", selection: $selectedModelId) {
                        Text("All Models").tag("all")
                        ForEach(presentation.modelOptions, id: \.self) { modelId in
                            Text(modelId).tag(modelId)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Workflow", selection: $selectedTaskRaw) {
                        Text("All Workflows").tag("all")
                        ForEach(presentation.taskOptions, id: \.rawValue) { task in
                            Text(task.title).tag(task.rawValue)
                        }
                    }
                    .pickerStyle(.menu)

                    Picker("Sort", selection: $selectedSort) {
                        ForEach(AssetSortMode.allCases, id: \.rawValue) { sort in
                            Text(sort.rawValue).tag(sort)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                GlassCard(title: "Collections", subtitle: "Keep working sets without losing the global library.") {
                    filterButton(
                        title: "All Assets",
                        isSelected: selectedCollectionId == nil,
                        count: presentation.count(for: nil)
                    ) {
                        selectedCollectionId = nil
                    }

                    ForEach(presentation.collections) { collection in
                        filterButton(
                            title: collection.title,
                            isSelected: selectedCollectionId == collection.id,
                            count: presentation.count(for: collection.id)
                        ) {
                            selectedCollectionId = collection.id
                        }
                    }

                    HStack(spacing: MLXRSpacing.xs) {
                        TextField("New collection", text: $newCollectionName)
                            .textFieldStyle(.roundedBorder)
                        Button("Add") {
                            let trimmed = newCollectionName.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            onCreateCollection(trimmed)
                            newCollectionName = ""
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            .padding(.horizontal, MLXRSpacing.lg)
            .padding(.vertical, MLXRSpacing.xl)
        }
    }

    private func filterSection<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            Text(title.uppercased())
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textTertiary)
            content()
        }
    }

    private func filterButton(
        title: String,
        isSelected: Bool,
        count: Int,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack {
                Text(title)
                Spacer()
                Text("\(count)")
                    .foregroundStyle(MLXRColor.textTertiary)
            }
            .font(MLXRType.bodySmall)
            .foregroundStyle(isSelected ? MLXRColor.brandPrimary : MLXRColor.textSecondary)
            .padding(.vertical, MLXRSpacing.xxs)
        }
        .buttonStyle(.plain)
    }
}
