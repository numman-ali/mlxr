import Foundation
import MLXRAppDomain

// MARK: - Built-In Recipes

/// The default set of recipes shipping with MLXR.
public enum BuiltInRecipes {

    public static let all: [Recipe] = [
        // ── Text to Image ──────────────────────────────────────────
        dreamscapeImage,
        portraitImage,
        landscapeImage,
        conceptArtImage,
        abstractImage,

        // ── Image Editing ──────────────────────────────────────────
        styleTransfer,
        imageRetouch,

        // ── Text to Video ──────────────────────────────────────────
        sceneVideo,
        cinematicVideo,
        abstractMotion,

        // ── Image to Video ─────────────────────────────────────────
        bringToLife,
        keyframeAnimation,

        // ── Motion Effects ─────────────────────────────────────────
        frameInterpolation,
        videoRetake,
        videoControl,

        // ── Audio Driven ───────────────────────────────────────────
        audioDrivenClip,
    ]

    // MARK: - Text to Image

    public static let dreamscapeImage = Recipe(
        id: "recipe.image.dreamscape",
        title: "Dreamscape",
        description: "Generate vivid, imaginative images from a text description. Great for concept exploration and creative ideation.",
        category: .textToImage,
        task: .imageGenerate,
        difficulty: .beginner,
        defaultModelFamily: "flux",
        examplePrompts: [
            "A bioluminescent forest at midnight with fireflies dancing around crystal mushrooms",
            "An astronaut floating through a nebula made of watercolors",
            "A steampunk clocktower growing out of an ancient tree, golden hour light",
        ],
        tags: ["creative", "imaginative", "starter"],
        defaultWidth: 1024,
        defaultHeight: 1024,
        defaultSteps: 4,
        defaultGuidance: 4.0,
        defaultArtifactFormat: "png"
    )

    public static let portraitImage = Recipe(
        id: "recipe.image.portrait",
        title: "Portrait Studio",
        description: "Create detailed character portraits with cinematic lighting. Optimized for faces and upper-body compositions.",
        category: .textToImage,
        task: .imageGenerate,
        difficulty: .beginner,
        defaultModelFamily: "flux",
        examplePrompts: [
            "A weathered sea captain with deep blue eyes, dramatic Rembrandt lighting",
            "A cyberpunk hacker with neon-lit face tattoos, moody purple atmosphere",
            "An elderly botanist surrounded by rare orchids, soft natural window light",
        ],
        tags: ["portraits", "characters", "lighting"],
        defaultWidth: 720,
        defaultHeight: 1280,
        defaultSteps: 20,
        defaultGuidance: 4.0,
        defaultArtifactFormat: "png"
    )

    public static let landscapeImage = Recipe(
        id: "recipe.image.landscape",
        title: "Landscape Vista",
        description: "Generate sweeping landscapes and environments in widescreen format. Perfect for scene setting and backgrounds.",
        category: .textToImage,
        task: .imageGenerate,
        difficulty: .beginner,
        defaultModelFamily: "flux",
        examplePrompts: [
            "A misty valley with a river winding through autumn foliage, aerial view",
            "An alien planet surface with crystalline formations under twin suns",
            "A coastal cliffside town at sunset with fishing boats in the harbor",
        ],
        tags: ["landscapes", "environments", "widescreen"],
        defaultWidth: 1280,
        defaultHeight: 720,
        defaultSteps: 20,
        defaultGuidance: 4.0,
        defaultArtifactFormat: "png"
    )

    public static let conceptArtImage = Recipe(
        id: "recipe.image.concept",
        title: "Concept Art",
        description: "Generate professional concept art and illustrations with high detail. Uses more inference steps for refined output.",
        category: .textToImage,
        task: .imageGenerate,
        difficulty: .intermediate,
        defaultModelFamily: "flux",
        examplePrompts: [
            "A massive space station interior, brutalist architecture, volumetric fog, concept art",
            "A fantasy marketplace with hovering lanterns and exotic creatures, detailed illustration",
            "A post-apocalyptic greenhouse city, lush vegetation reclaiming concrete, matte painting",
        ],
        tags: ["concept art", "illustration", "detailed"],
        defaultWidth: 1024,
        defaultHeight: 1024,
        defaultSteps: 40,
        defaultGuidance: 5.0,
        defaultArtifactFormat: "png"
    )

    public static let abstractImage = Recipe(
        id: "recipe.image.abstract",
        title: "Abstract Composition",
        description: "Create abstract and experimental imagery. Lower guidance allows more creative interpretation.",
        category: .textToImage,
        task: .imageGenerate,
        difficulty: .intermediate,
        defaultModelFamily: "flux",
        examplePrompts: [
            "Fluid dynamics of liquid metal and aurora borealis colors, macro photography",
            "Geometric fractals dissolving into organic coral structures",
            "Synesthetic visualization of a jazz saxophone solo in midnight blues",
        ],
        tags: ["abstract", "experimental", "artistic"],
        defaultWidth: 1024,
        defaultHeight: 1024,
        defaultSteps: 20,
        defaultGuidance: 2.5,
        defaultArtifactFormat: "png"
    )

    // MARK: - Image Editing

    public static let styleTransfer = Recipe(
        id: "recipe.edit.style-transfer",
        title: "Style Transfer",
        description: "Transform the visual style of an existing image while preserving its composition. Apply artistic treatments and mood changes.",
        category: .imageEditing,
        task: .imageEdit,
        difficulty: .beginner,
        defaultModelFamily: "flux",
        requiredReferences: [.image],
        examplePrompts: [
            "Transform into a watercolor painting with soft edges and muted tones",
            "Apply a cinematic color grade with teal shadows and warm highlights",
            "Render as a detailed pencil sketch with cross-hatching",
        ],
        tags: ["style", "transform", "artistic"],
        defaultWidth: 1024,
        defaultHeight: 1024,
        defaultSteps: 20,
        defaultGuidance: 4.0,
        defaultArtifactFormat: "png"
    )

    public static let imageRetouch = Recipe(
        id: "recipe.edit.retouch",
        title: "Image Retouch",
        description: "Make targeted edits to an existing image. Modify elements, fix issues, or add new details while keeping the rest intact.",
        category: .imageEditing,
        task: .imageEdit,
        difficulty: .intermediate,
        defaultModelFamily: "flux",
        requiredReferences: [.image],
        examplePrompts: [
            "Remove the background and replace with a starry night sky",
            "Change the lighting to golden hour with long warm shadows",
            "Add falling cherry blossom petals throughout the scene",
        ],
        tags: ["edit", "retouch", "modify"],
        defaultWidth: 1024,
        defaultHeight: 1024,
        defaultSteps: 20,
        defaultGuidance: 4.0,
        defaultArtifactFormat: "png"
    )

    // MARK: - Text to Video

    public static let sceneVideo = Recipe(
        id: "recipe.video.scene",
        title: "Scene in Motion",
        description: "Create a short video clip from a text description. Generates natural motion and camera movement.",
        category: .textToVideo,
        task: .videoGenerate,
        difficulty: .beginner,
        defaultModelFamily: "ltx-video",
        examplePrompts: [
            "A cat stretching lazily on a sunlit windowsill, dust motes floating in the air",
            "Waves crashing on a rocky shoreline at sunset, camera slowly panning right",
            "A candle flame flickering in a dark room, extreme close-up with shallow depth of field",
        ],
        tags: ["video", "motion", "scenes", "starter"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 30,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    public static let cinematicVideo = Recipe(
        id: "recipe.video.cinematic",
        title: "Cinematic Clip",
        description: "Generate longer, higher-quality video with cinematic motion. Uses more frames and steps for refined output.",
        category: .textToVideo,
        task: .videoGenerate,
        difficulty: .intermediate,
        defaultModelFamily: "ltx-video",
        examplePrompts: [
            "A drone shot sweeping over a misty mountain range at dawn, epic cinematic",
            "A slow-motion water droplet hitting a still pond, ripples expanding outward",
            "A time-lapse of storm clouds building over a city skyline, dramatic lighting",
        ],
        tags: ["cinematic", "high-quality", "dramatic"],
        defaultWidth: 848,
        defaultHeight: 480,
        defaultSteps: 50,
        defaultGuidance: 3.5,
        defaultArtifactFormat: "mp4"
    )

    public static let abstractMotion = Recipe(
        id: "recipe.video.abstract",
        title: "Abstract Motion",
        description: "Generate abstract and experimental motion art. Creates flowing, organic movement patterns.",
        category: .textToVideo,
        task: .videoGenerate,
        difficulty: .intermediate,
        defaultModelFamily: "ltx-video",
        examplePrompts: [
            "Liquid mercury flowing over iridescent surfaces, slow hypnotic movement",
            "Geometric shapes morphing and tessellating in a cosmic void",
            "Color gradients breathing and pulsing like living organisms",
        ],
        tags: ["abstract", "experimental", "art"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 2.0,
        defaultArtifactFormat: "mp4"
    )

    // MARK: - Image to Video

    public static let bringToLife = Recipe(
        id: "recipe.i2v.bring-to-life",
        title: "Bring to Life",
        description: "Animate a still image with natural motion. The image becomes the first frame of the video.",
        category: .imageToVideo,
        task: .videoConditionImage,
        difficulty: .beginner,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.image],
        examplePrompts: [
            "The subject slowly turns their head and smiles, gentle breeze in hair",
            "The scene comes alive with rustling leaves and drifting clouds",
            "Gentle camera push-in with subtle parallax motion",
        ],
        tags: ["animate", "image-to-video", "starter"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 30,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    public static let keyframeAnimation = Recipe(
        id: "recipe.i2v.keyframes",
        title: "Keyframe Animation",
        description: "Provide start and end frame images to generate the motion in between. Great for controlled transitions.",
        category: .imageToVideo,
        task: .videoInterpolate,
        difficulty: .advanced,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.image],
        optionalReferences: [.image],
        examplePrompts: [
            "Smooth morphing transition between the two frames",
            "Cinematic camera movement connecting both views",
            "Natural motion interpolation with realistic physics",
        ],
        tags: ["keyframes", "interpolation", "controlled"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    // MARK: - Motion Effects

    public static let frameInterpolation = Recipe(
        id: "recipe.fx.interpolate",
        title: "Frame Interpolation",
        description: "Generate smooth motion between keyframed images. Creates fluid transitions with natural in-between frames.",
        category: .motionEffects,
        task: .videoInterpolate,
        difficulty: .intermediate,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.image],
        examplePrompts: [
            "Interpolate with natural easing and realistic motion blur",
            "Create a smooth slow-motion transition between frames",
            "Generate physically plausible motion between the key poses",
        ],
        tags: ["interpolation", "slow-motion", "effects"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    public static let videoRetake = Recipe(
        id: "recipe.fx.retake",
        title: "Video Retake",
        description: "Regenerate a specific time window within an existing video clip. Keep the parts you like, redo the rest.",
        category: .motionEffects,
        task: .videoRetake,
        difficulty: .advanced,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.video],
        examplePrompts: [
            "Redo this section with more dramatic camera movement",
            "Regenerate with different lighting and mood",
            "Replace this window with smoother, more natural motion",
        ],
        tags: ["retake", "regenerate", "edit"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    public static let videoControl = Recipe(
        id: "recipe.fx.control",
        title: "Video Control",
        description: "Condition generation with a source video and optional LoRA control. Apply effects and transformations guided by existing motion.",
        category: .motionEffects,
        task: .videoConditionVideo,
        difficulty: .advanced,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.video],
        optionalReferences: [.lora],
        examplePrompts: [
            "Apply this motion to a new scene: a futuristic city at night",
            "Transfer the camera movement to an underwater coral reef scene",
            "Restyle the video as an oil painting with impressionist brush strokes",
        ],
        tags: ["control", "condition", "advanced"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )

    // MARK: - Audio Driven

    public static let audioDrivenClip = Recipe(
        id: "recipe.audio.driven",
        title: "Audio-Driven Video",
        description: "Generate video motion conditioned on audio timing and energy. The clip's movement responds to the music or sound.",
        category: .audioDriven,
        task: .videoConditionAudio,
        difficulty: .intermediate,
        defaultModelFamily: "ltx-video",
        requiredReferences: [.audio],
        examplePrompts: [
            "Abstract visuals that pulse and flow with the beat of the music",
            "A surreal landscape that breathes and shifts with the audio energy",
            "Particle systems and light trails synced to the rhythm",
        ],
        tags: ["audio", "music", "reactive"],
        defaultWidth: 704,
        defaultHeight: 704,
        defaultSteps: 40,
        defaultGuidance: 3.0,
        defaultArtifactFormat: "mp4"
    )
}
