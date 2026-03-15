@preconcurrency import AppKit
import Foundation
import ImageIO

@MainActor
public enum PreviewImageLoader {
    private static let imageCache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 48
        return cache
    }()

    private static var dominantHueCache: [String: CGFloat] = [:]
    private static var inflight: [String: Task<(NSImage?, CGFloat?), Never>] = [:]

    public static func load(
        from url: URL,
        maxPixelSize: CGFloat = 2_048
    ) async -> (NSImage?, CGFloat?) {
        let cacheKey = "\(url.path(percentEncoded: false))::\(Int(maxPixelSize.rounded()))"
        let nsCacheKey = cacheKey as NSString
        if let cachedImage = imageCache.object(forKey: nsCacheKey) {
            return (cachedImage, dominantHueCache[cacheKey])
        }
        if let task = inflight[cacheKey] {
            return await task.value
        }

        let task = Task.detached(priority: .userInitiated) { () -> (NSImage?, CGFloat?) in
            guard let image = downsampledImage(at: url, maxPixelSize: maxPixelSize) else {
                return (nil, nil)
            }
            return (image, dominantHue(from: image))
        }
        inflight[cacheKey] = task
        let preview = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        inflight.removeValue(forKey: cacheKey)
        if let image = preview.0 {
            imageCache.setObject(image, forKey: nsCacheKey)
            dominantHueCache[cacheKey] = preview.1
        } else {
            dominantHueCache.removeValue(forKey: cacheKey)
        }
        return preview
    }

    nonisolated private static func downsampledImage(
        at url: URL,
        maxPixelSize: CGFloat
    ) -> NSImage? {
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
