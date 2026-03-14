import AVKit
import SwiftUI

// MARK: - Media Thumbnail

public struct MediaThumbnail: View {
    private let image: NSImage?
    private let aspectRatio: CGFloat

    public init(image: NSImage? = nil, aspectRatio: CGFloat = 1.0) {
        self.image = image
        self.aspectRatio = aspectRatio
    }

    public var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(aspectRatio, contentMode: .fill)
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous)
                        .fill(MLXRColor.surfaceOverlay)

                    Image(systemName: "photo")
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(MLXRColor.textTertiary)
                }
                .aspectRatio(aspectRatio, contentMode: .fill)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.md, style: .continuous))
    }
}

// MARK: - Video Thumbnail

public struct VideoThumbnail: View {
    private let image: NSImage?
    private let duration: String?

    public init(image: NSImage? = nil, duration: String? = nil) {
        self.image = image
        self.duration = duration
    }

    public var body: some View {
        ZStack(alignment: .bottomTrailing) {
            MediaThumbnail(image: image, aspectRatio: 16.0 / 9.0)

            // Play indicator
            ZStack {
                Circle()
                    .fill(.black.opacity(0.4))
                    .frame(width: 40, height: 40)
                Image(systemName: "play.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Duration badge
            if let duration {
                Text(duration)
                    .font(MLXRType.captionSmall)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(.black.opacity(0.6)))
                    .padding(MLXRSpacing.xs)
            }
        }
    }
}

// MARK: - Media Hero (large preview with adaptive glow)

public struct MediaHero: View {
    private let image: NSImage?
    private let dominantHue: CGFloat?

    public init(image: NSImage? = nil, dominantHue: CGFloat? = nil) {
        self.image = image
        self.dominantHue = dominantHue
    }

    public var body: some View {
        ZStack {
            // Ambient glow background
            if let hue = dominantHue {
                RoundedRectangle(cornerRadius: MLXRRadius.xl, style: .continuous)
                    .fill(MLXRColor.adaptiveTint(from: hue, saturation: 0.4, opacity: 0.2))
                    .blur(radius: 40)
                    .scaleEffect(1.1)
            }

            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous))
                    .shadow(color: .black.opacity(0.3), radius: 20, y: 8)
            } else {
                MediaThumbnail(aspectRatio: 16.0 / 9.0)
            }
        }
    }
}

// MARK: - Generation Card (gallery card for running/completed job)

public struct GenerationCard: View {
    public enum CardState: Sendable {
        case generating(progress: Double, phase: String?)
        case completed(image: NSImage?)
        case failed(message: String)
        case cancelled
    }

    private let state: CardState
    private let prompt: String
    private let taskLabel: String?

    public init(state: CardState, prompt: String, taskLabel: String? = nil) {
        self.state = state
        self.prompt = prompt
        self.taskLabel = taskLabel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Media area
            ZStack {
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.04), Color.white.opacity(0.02)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .aspectRatio(1.0, contentMode: .fill)

                switch state {
                case .generating(let progress, let phase):
                    VStack(spacing: MLXRSpacing.sm) {
                        MLXRProgressRing(progress: progress, size: 56)
                        if let phase {
                            Text(phase)
                                .font(MLXRType.captionSmall)
                                .foregroundStyle(MLXRColor.textTertiary)
                                .lineLimit(1)
                        }
                    }

                case .completed(let image):
                    if let image {
                        Image(nsImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    }

                case .failed:
                    VStack(spacing: MLXRSpacing.xs) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(MLXRColor.brandDanger)
                        Text("Failed")
                            .font(MLXRType.captionSmall)
                            .foregroundStyle(MLXRColor.brandDanger.opacity(0.7))
                    }

                case .cancelled:
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(MLXRColor.textTertiary)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [Color.white.opacity(0.1), Color.white.opacity(0.03)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.5
                    )
            )

            // Info area
            VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                Text(prompt)
                    .font(MLXRType.bodySmall)
                    .foregroundStyle(MLXRColor.textSecondary)
                    .lineLimit(2)

                if let taskLabel {
                    StatusPill(label: taskLabel, tint: MLXRColor.brandPrimary)
                }
            }
            .padding(.top, MLXRSpacing.sm)
        }
    }
}

// MARK: - Progress Ring

public struct MLXRProgressRing: View {
    private let progress: Double
    private let lineWidth: CGFloat
    private let size: CGFloat

    public init(progress: Double, lineWidth: CGFloat = 3, size: CGFloat = 48) {
        self.progress = min(max(progress, 0), 1)
        self.lineWidth = lineWidth
        self.size = size
    }

    public var body: some View {
        ZStack {
            Circle()
                .strokeBorder(MLXRColor.surfaceOverlay, lineWidth: lineWidth)

            Circle()
                .trim(from: 0, to: progress)
                .stroke(
                    AngularGradient(
                        colors: [MLXRColor.brandPrimary, MLXRColor.brandSecondary],
                        center: .center
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))
                .animation(MLXRMotion.spring, value: progress)

            Text("\(Int(progress * 100))")
                .font(MLXRType.captionSmall)
                .foregroundStyle(MLXRColor.textSecondary)
                .contentTransition(.numericText())
        }
        .frame(width: size, height: size)
    }
}

// MARK: - Dominant Color Extraction

public func dominantHue(from image: NSImage) -> CGFloat? {
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return nil
    }

    let width = 16
    let height = 16
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

    guard let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return nil
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    var totalR: CGFloat = 0, totalG: CGFloat = 0, totalB: CGFloat = 0
    let pixelCount = CGFloat(width * height)

    for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
        totalR += CGFloat(pixelData[i]) / 255.0
        totalG += CGFloat(pixelData[i + 1]) / 255.0
        totalB += CGFloat(pixelData[i + 2]) / 255.0
    }

    let avgR = totalR / pixelCount
    let avgG = totalG / pixelCount
    let avgB = totalB / pixelCount

    let maxC = max(avgR, avgG, avgB)
    let minC = min(avgR, avgG, avgB)
    let delta = maxC - minC

    guard delta > 0.05 else { return nil }

    var hue: CGFloat
    if maxC == avgR {
        hue = ((avgG - avgB) / delta).truncatingRemainder(dividingBy: 6)
    } else if maxC == avgG {
        hue = (avgB - avgR) / delta + 2
    } else {
        hue = (avgR - avgG) / delta + 4
    }
    hue /= 6
    if hue < 0 { hue += 1 }
    return hue
}
