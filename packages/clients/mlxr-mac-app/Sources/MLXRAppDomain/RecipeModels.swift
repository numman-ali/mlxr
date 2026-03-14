import Foundation

// MARK: - Destination (replaces ProductSurface)

public enum Destination: String, CaseIterable, Identifiable, Sendable, Codable {
    case home
    case studio
    case library
    case models
    case settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .home: "Home"
        case .studio: "Studio"
        case .library: "Library"
        case .models: "Models"
        case .settings: "Settings"
        }
    }

    public var icon: String {
        switch self {
        case .home: "house.fill"
        case .studio: "sparkles.rectangle.stack.fill"
        case .library: "photo.stack.fill"
        case .models: "square.stack.3d.up.fill"
        case .settings: "gearshape.fill"
        }
    }
}

// MARK: - Recipe

public struct Recipe: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let description: String
    public let category: RecipeCategory
    public let task: ProductTask
    public let difficulty: RecipeDifficulty
    public let defaultModelFamily: String?
    public let requiredReferences: [WorkflowReferenceKind]
    public let optionalReferences: [WorkflowReferenceKind]
    public let examplePrompts: [String]
    public let tags: [String]
    public let defaultWidth: Int?
    public let defaultHeight: Int?
    public let defaultSteps: Int?
    public let defaultGuidance: Double?
    public let defaultArtifactFormat: String?

    public init(
        id: String,
        title: String,
        description: String,
        category: RecipeCategory,
        task: ProductTask,
        difficulty: RecipeDifficulty = .beginner,
        defaultModelFamily: String? = nil,
        requiredReferences: [WorkflowReferenceKind] = [],
        optionalReferences: [WorkflowReferenceKind] = [],
        examplePrompts: [String] = [],
        tags: [String] = [],
        defaultWidth: Int? = nil,
        defaultHeight: Int? = nil,
        defaultSteps: Int? = nil,
        defaultGuidance: Double? = nil,
        defaultArtifactFormat: String? = nil
    ) {
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.task = task
        self.difficulty = difficulty
        self.defaultModelFamily = defaultModelFamily
        self.requiredReferences = requiredReferences
        self.optionalReferences = optionalReferences
        self.examplePrompts = examplePrompts
        self.tags = tags
        self.defaultWidth = defaultWidth
        self.defaultHeight = defaultHeight
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.defaultArtifactFormat = defaultArtifactFormat
    }

    /// Whether this recipe requires any file reference inputs
    public var needsReferences: Bool {
        !requiredReferences.isEmpty
    }

    /// Filter label for the recipe browser
    public var filterCategory: RecipeFilterCategory {
        switch category {
        case .textToImage, .imageEditing:
            .images
        case .textToVideo, .imageToVideo, .motionEffects, .audioDriven:
            .video
        }
    }
}

public enum RecipeCategory: String, CaseIterable, Sendable, Codable {
    case textToImage
    case imageEditing
    case textToVideo
    case imageToVideo
    case motionEffects
    case audioDriven
}

public enum RecipeDifficulty: String, CaseIterable, Sendable, Codable {
    case beginner
    case intermediate
    case advanced
}

public enum RecipeFilterCategory: String, CaseIterable, Sendable, Codable {
    case all = "All"
    case images = "Images"
    case video = "Video"
    case editing = "Editing"
}

// MARK: - Cold Start Phase

public enum ColdStartPhase: Sendable {
    case loading
    case showcase       // No models installed
    case firstCreate    // Models installed but no outputs
    case complete       // User has outputs
}

// MARK: - Quality Preset (human-readable)

public enum QualityPreset: String, CaseIterable, Sendable, Codable {
    case draft = "Draft"
    case standard = "Standard"
    case cinematic = "Cinematic"

    public func steps(for category: TaskCategory) -> Int {
        switch (self, category) {
        case (.draft, .image): 4
        case (.standard, .image): 20
        case (.cinematic, .image): 40
        case (.draft, .video): 15
        case (.standard, .video): 30
        case (.cinematic, .video): 50
        }
    }

    public var quality: WorkflowQuality {
        switch self {
        case .draft: .fast
        case .standard: .balanced
        case .cinematic: .high
        }
    }
}

// MARK: - Aspect Preset

public enum AspectPreset: String, CaseIterable, Sendable, Codable {
    case square = "Square"
    case landscape = "Landscape"
    case portrait = "Portrait"
    case story = "Story"

    public func dimensions(for category: TaskCategory) -> (width: Int, height: Int) {
        switch (self, category) {
        case (.square, .image): (1024, 1024)
        case (.landscape, .image): (1280, 720)
        case (.portrait, .image): (720, 1280)
        case (.story, .image): (1080, 1920)
        case (.square, .video): (704, 704)
        case (.landscape, .video): (848, 480)
        case (.portrait, .video): (480, 848)
        case (.story, .video): (540, 960)
        }
    }
}

// MARK: - Duration Preset (video only)

public enum DurationPreset: String, CaseIterable, Sendable, Codable {
    case short = "Short"
    case medium = "Medium"
    case long = "Long"

    public var numFrames: Int {
        switch self {
        case .short: 41
        case .medium: 81
        case .long: 161
        }
    }

    public var approximateSeconds: String {
        switch self {
        case .short: "~2s"
        case .medium: "~3s"
        case .long: "~7s"
        }
    }
}
