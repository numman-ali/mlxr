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
        .library(name: "MLXRRuntimeBridge", targets: ["MLXRRuntimeBridge"]),
        .library(name: "MLXRFeatureImages", targets: ["MLXRFeatureImages"]),
        .library(name: "MLXRFeatureVideo", targets: ["MLXRFeatureVideo"]),
        .library(name: "MLXRFeatureModels", targets: ["MLXRFeatureModels"]),
        .library(name: "MLXRFeatureLibrary", targets: ["MLXRFeatureLibrary"]),
        .library(name: "MLXRFeatureJobs", targets: ["MLXRFeatureJobs"]),
        .library(name: "MLXRFeatureSettings", targets: ["MLXRFeatureSettings"]),
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
            name: "MLXRFeatureImages",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureVideo",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureModels",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureLibrary",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureJobs",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRFeatureSettings",
            dependencies: ["MLXRDesignSystem", "MLXRAppDomain", "MLXRRuntimeBridge"]
        ),
        .target(
            name: "MLXRAppShell",
            dependencies: [
                "MLXRDesignSystem",
                "MLXRAppDomain",
                "MLXRRuntimeBridge",
                "MLXRFeatureImages",
                "MLXRFeatureVideo",
                "MLXRFeatureModels",
                "MLXRFeatureLibrary",
                "MLXRFeatureJobs",
                "MLXRFeatureSettings",
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
