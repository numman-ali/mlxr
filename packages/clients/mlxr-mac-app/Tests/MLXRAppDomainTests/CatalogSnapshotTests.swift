import Testing
@testable import MLXRAppDomain

@Test
func catalogMergesSupportedAndInstalledModels() {
    let supported = SupportedModelDescriptor(
        modelId: "flux2-klein-9b-local",
        displayName: "FLUX.2 Klein 9B",
        family: "flux2",
        familyVariant: "flux.2-klein-9b",
        recommendationTier: .recommended,
        supportLevel: .promoted,
        tasks: ["image.generate", "image.edit"],
        provider: "huggingface",
        sourceSummary: "black-forest-labs/FLUX.2-klein-9B",
        license: "other-non-commercial",
        accessState: "public",
        installed: true,
        installable: true,
        notes: nil
    )
    let capability = CapabilityDescriptor(
        modelId: "flux2-klein-9b-local",
        artifactDigest: "sha256:test",
        family: "flux2",
        familyVariant: "flux.2-klein-9b",
        tasks: ["image.generate", "image.edit"],
        modalitiesIn: ["text", "image"],
        modalitiesOut: ["image"],
        constraints: [:],
        conditioning: ["image": .bool(true)],
        profilesByTask: [:],
        streaming: [:],
        artifactsOut: ["png"],
        schedulerClass: "image_diffusion",
        hardwareTiers: [],
        dependencies: [:],
        policy: PolicyDescriptor(
            license: "other-non-commercial",
            accessState: "public",
            remoteCodeRequired: false,
            remoteCodeApproved: false,
            redistributionState: nil
        ),
        extensionsSchema: nil,
        metadata: [:]
    )
    let installed = ModelRecord(
        modelId: "flux2-klein-9b-local",
        family: "flux2",
        source: nil,
        artifact: nil,
        loaded: true,
        capability: capability
    )

    let snapshot = CatalogSnapshot(
        supportedModels: [supported],
        installedModels: [installed],
        capabilities: [capability]
    )

    #expect(snapshot.items.count == 1)
    #expect(snapshot.defaultModel(for: .imageGenerate)?.modelId == "flux2-klein-9b-local")
    #expect(snapshot.items[0].installed)
    #expect(snapshot.items[0].isRecommended)
    #expect(snapshot.items[0].tasks == [.imageGenerate, .imageEdit])
}

@Test
func catalogIncludesInstalledAdvancedRowsOutsideCuratedCatalog() {
    let capability = CapabilityDescriptor(
        modelId: "ltx-dev-two-stage",
        artifactDigest: "sha256:dev",
        family: "ltx",
        familyVariant: "dev",
        tasks: ["video.generate", "video.interpolate"],
        modalitiesIn: ["text", "image"],
        modalitiesOut: ["video"],
        constraints: [:],
        conditioning: ["image": .bool(true)],
        profilesByTask: [:],
        streaming: [:],
        artifactsOut: ["mp4"],
        schedulerClass: "media_video_dit",
        hardwareTiers: [],
        dependencies: [:],
        policy: PolicyDescriptor(
            license: "other",
            accessState: "gated",
            remoteCodeRequired: false,
            remoteCodeApproved: false,
            redistributionState: nil
        ),
        extensionsSchema: nil,
        metadata: [
            "implemented_surface": .object([
                "pipeline_variants": .array([.string("one_stage"), .string("two_stage")]),
            ]),
        ]
    )
    let installed = ModelRecord(
        modelId: "ltx-dev-two-stage",
        family: "ltx",
        source: nil,
        artifact: nil,
        loaded: false,
        capability: capability
    )

    let snapshot = CatalogSnapshot(
        supportedModels: [],
        installedModels: [installed],
        capabilities: [capability]
    )

    #expect(snapshot.items.count == 1)
    #expect(snapshot.items[0].recommendationTier == .advanced)
    #expect(snapshot.items[0].implementedPipelineVariants == ["one_stage", "two_stage"])
    #expect(snapshot.items(for: .videoInterpolate).first?.modelId == "ltx-dev-two-stage")
}

@Test
func catalogPreservesSupportedOrderingForDefaultRecommendations() {
    let supported: [SupportedModelDescriptor] = [
        SupportedModelDescriptor(
            modelId: "z-image-turbo-local",
            displayName: "Z-Image Turbo",
            family: "z_image",
            familyVariant: "z-image-turbo",
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "huggingface",
            sourceSummary: "Tongyi-MAI/Z-Image-Turbo",
            license: "other",
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        ),
        SupportedModelDescriptor(
            modelId: "qwen-image-local",
            displayName: "Qwen-Image 2512",
            family: "qwen_image",
            familyVariant: "qwen-image-2512",
            recommendationTier: .recommended,
            supportLevel: .promoted,
            tasks: ["image.generate"],
            provider: "huggingface",
            sourceSummary: "Qwen/Qwen-Image-2512",
            license: "apache-2.0",
            accessState: "public",
            installed: true,
            installable: true,
            notes: nil
        ),
    ]

    let snapshot = CatalogSnapshot(
        supportedModels: supported,
        installedModels: [],
        capabilities: []
    )

    #expect(snapshot.items.map(\.modelId) == ["z-image-turbo-local", "qwen-image-local"])
    #expect(snapshot.defaultModel(for: .imageGenerate)?.modelId == "z-image-turbo-local")
}
