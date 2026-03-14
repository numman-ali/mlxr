import AppKit
import SwiftUI

// MARK: - Color System (Dark-Cinematic First)

public enum MLXRColor {

    // -- Canvas backgrounds --

    public static let canvasDeep = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.96, green: 0.97, blue: 0.98, alpha: 1),
        dark: .init(srgbRed: 0.03, green: 0.03, blue: 0.06, alpha: 1)
    ))

    public static let canvasRaised = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.98, green: 0.98, blue: 0.99, alpha: 1),
        dark: .init(srgbRed: 0.07, green: 0.07, blue: 0.10, alpha: 1)
    ))

    // -- Surface layers --

    public static let surfaceCard = Color(nsColor: adaptive(
        light: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.72),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.06)
    ))

    public static let surfaceHover = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.03),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.10)
    ))

    public static let surfaceOverlay = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.04),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.04)
    ))

    // -- Border --

    public static let borderSubtle = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.08),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.10)
    ))

    public static let borderFocused = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.10, green: 0.48, blue: 0.96, alpha: 0.4),
        dark: .init(srgbRed: 0.38, green: 0.64, blue: 1.0, alpha: 0.4)
    ))

    // -- Brand accents --

    public static let brandPrimary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.10, green: 0.48, blue: 0.96, alpha: 1),
        dark: .init(srgbRed: 0.38, green: 0.64, blue: 1.0, alpha: 1)
    ))

    public static let brandSecondary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.00, green: 0.73, blue: 0.66, alpha: 1),
        dark: .init(srgbRed: 0.24, green: 0.86, blue: 0.71, alpha: 1)
    ))

    public static let brandWarm = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.95, green: 0.55, blue: 0.18, alpha: 1),
        dark: .init(srgbRed: 1.0, green: 0.70, blue: 0.31, alpha: 1)
    ))

    public static let brandDanger = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.96, green: 0.26, blue: 0.21, alpha: 1),
        dark: .init(srgbRed: 1.0, green: 0.41, blue: 0.38, alpha: 1)
    ))

    public static let brandGlow = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.10, green: 0.48, blue: 0.96, alpha: 0.15),
        dark: .init(srgbRed: 0.38, green: 0.64, blue: 1.0, alpha: 0.20)
    ))

    // -- Text hierarchy --

    public static let textPrimary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.88),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.92)
    ))

    public static let textSecondary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.55),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.60)
    ))

    public static let textTertiary = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.32),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.38)
    ))

    public static let textDisabled = Color(nsColor: adaptive(
        light: .init(srgbRed: 0.0, green: 0.0, blue: 0.0, alpha: 0.20),
        dark: .init(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 0.24)
    ))

    // -- Gradients --

    public static let heroGradient = LinearGradient(
        colors: [
            Color(red: 0.05, green: 0.10, blue: 0.25),
            Color(red: 0.03, green: 0.26, blue: 0.45),
            Color(red: 0.06, green: 0.40, blue: 0.36),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    public static let brandGradient = LinearGradient(
        colors: [
            Color(red: 0.20, green: 0.45, blue: 0.95),
            Color(red: 0.10, green: 0.65, blue: 0.80),
        ],
        startPoint: .leading,
        endPoint: .trailing
    )

    public static let canvasGradient = LinearGradient(
        colors: [
            Color(red: 0.04, green: 0.04, blue: 0.07),
            Color(red: 0.03, green: 0.03, blue: 0.06),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    public static let shimmerGradient = LinearGradient(
        colors: [.clear, .white.opacity(0.08), .clear],
        startPoint: .leading,
        endPoint: .trailing
    )

    public static func cinematicGradient(hue: CGFloat) -> LinearGradient {
        LinearGradient(
            colors: [
                Color(hue: hue, saturation: 0.6, brightness: 0.15),
                Color(hue: hue, saturation: 0.4, brightness: 0.08),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    /// Content-adaptive tint color from a dominant hue
    public static func adaptiveTint(from hue: CGFloat, saturation: CGFloat = 0.5, opacity: CGFloat = 0.15) -> Color {
        Color(hue: hue, saturation: saturation, brightness: 0.6).opacity(opacity)
    }

    // -- Helpers --

    private static func adaptive(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        }
    }
}

// MARK: - Legacy Compatibility (remove after full migration)

public typealias MLXRTheme = MLXRColor

extension MLXRColor {
    public static let accent = brandPrimary
    public static let secondaryAccent = brandSecondary
    public static let warmAccent = brandWarm
    public static let destructive = brandDanger
    public static let surfacePrimary = surfaceCard
    public static let surfaceSecondary = surfaceOverlay

    public static let backgroundGradientLight = LinearGradient(
        colors: [
            Color(red: 0.96, green: 0.97, blue: 1.0),
            Color(red: 0.93, green: 0.96, blue: 0.98),
        ],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    public static let backgroundGradientDark = canvasGradient
}
