import SwiftUI

// MARK: - Type Scale

public enum MLXRType {
    public static let displayLarge = Font.system(size: 42, weight: .bold, design: .rounded)
    public static let displayMedium = Font.system(size: 32, weight: .bold, design: .rounded)
    public static let titleLarge = Font.system(size: 24, weight: .semibold, design: .rounded)
    public static let titleMedium = Font.system(size: 20, weight: .semibold, design: .rounded)
    public static let titleSmall = Font.system(size: 17, weight: .medium, design: .rounded)
    public static let bodyLarge = Font.system(size: 16, weight: .regular, design: .rounded)
    public static let bodyMedium = Font.system(size: 14, weight: .regular, design: .rounded)
    public static let bodySmall = Font.system(size: 13, weight: .regular, design: .rounded)
    public static let captionLarge = Font.system(size: 12, weight: .medium, design: .rounded)
    public static let captionSmall = Font.system(size: 11, weight: .semibold, design: .rounded)
    public static let monoSmall = Font.system(size: 12, weight: .medium, design: .monospaced)
}

// MARK: - Text Modifiers

public struct EyebrowStyle: ViewModifier {
    public func body(content: Content) -> some View {
        content
            .font(MLXRType.captionSmall)
            .tracking(1.2)
            .textCase(.uppercase)
            .foregroundStyle(MLXRColor.textTertiary)
    }
}

extension View {
    public func eyebrowStyle() -> some View {
        modifier(EyebrowStyle())
    }
}
