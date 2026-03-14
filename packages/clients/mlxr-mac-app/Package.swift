// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlxr-mac-app",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "MLXRDesignSystem", targets: ["MLXRDesignSystem"]),
        .library(name: "MLXRAppDomain", targets: ["MLXRAppDomain"]),
        .library(name: "MLXRRecipes", targets: ["MLXRRecipes"]),
        .library(name: "MLXRRuntimeBridge", targets: ["MLXRRuntimeBridge"]),
        .library(name: "MLXRFeatureHome", targets: ["MLXRFeatureHome"]),
        .library(name: "MLXRFeatureCreate", targets: ["MLXRFeatureCreate"]),
        .library(name: "MLXRFeatureGallery", targets: ["MLXRFeatureGallery"]),
        .library(name: "MLXRFeatureToolkit", targets: ["MLXRFeatureToolkit"]),
        .library(name: "MLXRFeatureSettings", targets: ["MLXRFeatureSettings"]),
        .library(name: "MLXRActivityStrip", targets: ["MLXRActivityStrip"]),
        .library(name: "MLXRAppShell", targets: ["MLXRAppShell"]),
        .executable(name: "MLXRMacApp", targets: ["MLXRMacApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.30.0"),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.80.0"),
        .package(url: "https://github.com/apple/swift-testing.git", exact: "6.2.1"),
    ],
    targets: [
        .target(name: "MLXRDesignSystem"),
        .target(name: "MLXRAppDomain"),
        .target(
            name: "MLXRRecipes",
            dependencies: ["MLXRAppDomain"]
        ),
        .target(
            name: "MLXRRuntimeBridge",
            dependencies: [
                "MLXRAppDomain",
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOFoundationCompat", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .target(
            name: "MLXRFeatureHome",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRecipes", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureCreate",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRecipes", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureGallery",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureToolkit",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureSettings",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRActivityStrip",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRAppShell",
            dependencies: [
                "MLXRDesignSystem",
                "MLXRAppDomain",
                "MLXRRecipes",
                "MLXRRuntimeBridge",
                "MLXRFeatureHome",
                "MLXRFeatureCreate",
                "MLXRFeatureGallery",
                "MLXRFeatureToolkit",
                "MLXRFeatureSettings",
                "MLXRActivityStrip",
            ]
        ),
        .executableTarget(name: "MLXRMacApp", dependencies: ["MLXRAppShell"]),
        .testTarget(
            name: "MLXRAppDomainTests",
            dependencies: [
                "MLXRAppDomain",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
        .testTarget(
            name: "MLXRRuntimeBridgeTests",
            dependencies: [
                "MLXRAppDomain",
                "MLXRRuntimeBridge",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
        .testTarget(
            name: "MLXRAppShellTests",
            dependencies: [
                "MLXRAppDomain",
                "MLXRRuntimeBridge",
                "MLXRAppShell",
                .product(name: "Testing", package: "swift-testing"),
            ]
        ),
    ]
)
