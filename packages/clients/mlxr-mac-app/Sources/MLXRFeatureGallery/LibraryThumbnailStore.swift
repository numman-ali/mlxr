import AppKit
import AVFoundation
import ImageIO
import MLXRAppDomain
import Foundation

@MainActor
final class LibraryThumbnailStore {
    static let shared = LibraryThumbnailStore()

    private let cache = NSCache<NSString, NSImage>()
    private var inflight: [String: Task<NSImage?, Never>] = [:]

    private init() {
        cache.countLimit = 512
    }

    func thumbnail(
        for asset: LibraryAsset,
        maxPixelSize: CGFloat,
        materialize: @escaping @Sendable (LibraryAsset) async -> URL?
    ) async -> NSImage? {
        let cacheKey = "\(asset.id)::\(Int(maxPixelSize))" as NSString
        if let cached = cache.object(forKey: cacheKey) {
            return cached
        }
        let inflightKey = String(cacheKey)
        if let task = inflight[inflightKey] {
            return await task.value
        }

        let task = Task<NSImage?, Never> {
            guard let url = await materialize(asset) else { return nil }
            return await Self.makeThumbnail(
                for: asset,
                url: url,
                maxPixelSize: maxPixelSize
            )
        }
        inflight[inflightKey] = task
        let image = await task.value
        inflight.removeValue(forKey: inflightKey)
        if let image {
            cache.setObject(image, forKey: cacheKey)
        }
        return image
    }

    private static func makeThumbnail(
        for asset: LibraryAsset,
        url: URL,
        maxPixelSize: CGFloat
    ) async -> NSImage? {
        switch asset.kind {
        case .image:
            return downsampledImage(at: url, maxPixelSize: maxPixelSize)
        case .video:
            let avAsset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: avAsset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: maxPixelSize, height: maxPixelSize)
            let time = NSValue(time: .zero)
            return await withCheckedContinuation { continuation in
                generator.generateCGImagesAsynchronously(forTimes: [time]) { _, cgImage, _, _, _ in
                    if let cgImage {
                        continuation.resume(returning: NSImage(cgImage: cgImage, size: .zero))
                    } else {
                        continuation.resume(returning: nil)
                    }
                }
            }
        case .audio, .other:
            return nil
        }
    }

    private static func downsampledImage(at url: URL, maxPixelSize: CGFloat) -> NSImage? {
        guard
            let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateThumbnailAtIndex(
                imageSource,
                0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceShouldCacheImmediately: false,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: Int(maxPixelSize),
                ] as CFDictionary
            )
        else {
            return nil
        }
        return NSImage(cgImage: cgImage, size: .zero)
    }
}
