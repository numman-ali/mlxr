import Foundation
import MLXRAppDomain

// MARK: - Recipe Catalog

/// Filters available recipes based on installed model capabilities.
public struct RecipeCatalog: Sendable {

    public let allRecipes: [Recipe]

    public init(recipes: [Recipe] = BuiltInRecipes.all) {
        self.allRecipes = recipes
    }

    /// Returns all recipes whose task is supported by at least one model in the catalog.
    public func availableRecipes(for catalog: CatalogSnapshot) -> [Recipe] {
        allRecipes.filter { recipe in
            !catalog.items(for: recipe.task).isEmpty
        }
    }

    /// Returns recipes whose task has an installed model ready to run.
    public func readyRecipes(for catalog: CatalogSnapshot) -> [Recipe] {
        allRecipes.filter { recipe in
            catalog.items(for: recipe.task).contains(where: \.installed)
        }
    }

    /// Returns recipes grouped by filter category.
    public func grouped(for catalog: CatalogSnapshot) -> [RecipeFilterCategory: [Recipe]] {
        var result: [RecipeFilterCategory: [Recipe]] = [:]
        for category in RecipeFilterCategory.allCases {
            result[category] = availableRecipes(for: catalog).filter { recipe in
                recipe.filterCategory == category || category == .all
            }
        }
        return result
    }

    /// Returns the best recipe for a given task, preferring beginner difficulty.
    public func defaultRecipe(for task: ProductTask) -> Recipe? {
        let matching = allRecipes.filter { $0.task == task }
        return matching.first { $0.difficulty == .beginner }
            ?? matching.first
    }

    /// Searches recipes by query against title, description, and tags.
    public func search(query: String) -> [Recipe] {
        guard !query.isEmpty else { return allRecipes }
        let lowered = query.lowercased()
        return allRecipes.filter { recipe in
            recipe.title.lowercased().contains(lowered) ||
            recipe.description.lowercased().contains(lowered) ||
            recipe.tags.contains { $0.lowercased().contains(lowered) }
        }
    }

    /// Returns recommended recipes for a first-run experience.
    public func starterRecipes(for catalog: CatalogSnapshot) -> [Recipe] {
        let ready = readyRecipes(for: catalog)
        // One from each major category if available
        var starters: [Recipe] = []
        if let img = ready.first(where: { $0.category == .textToImage }) {
            starters.append(img)
        }
        if let vid = ready.first(where: { $0.category == .textToVideo }) {
            starters.append(vid)
        }
        if let i2v = ready.first(where: { $0.category == .imageToVideo }) {
            starters.append(i2v)
        }
        if let edit = ready.first(where: { $0.category == .imageEditing }) {
            starters.append(edit)
        }
        return starters
    }
}
