import SwiftUI

// MARK: - Animation Presets

public enum MLXRMotion {

    // -- Core springs (kept from original) --

    public static let spring = Animation.spring(response: 0.45, dampingFraction: 0.82)
    public static let snappy = Animation.spring(response: 0.3, dampingFraction: 0.78)
    public static let gentle = Animation.spring(response: 0.6, dampingFraction: 0.86)
    public static let quick = Animation.easeInOut(duration: 0.2)

    // -- New motion vocabulary --

    public static let micro = Animation.easeOut(duration: 0.15)
    public static let cinematic = Animation.spring(response: 0.7, dampingFraction: 0.90)

    public static let cardHover = Animation.spring(response: 0.3, dampingFraction: 0.80)
    public static let cardPress = Animation.spring(response: 0.2, dampingFraction: 0.70)

    public static let shimmer = Animation.easeInOut(duration: 1.5).repeatForever(autoreverses: false)

    public static func stagger(index: Int, base: Animation = .spring(response: 0.5, dampingFraction: 0.82)) -> Animation {
        base.delay(Double(index) * 0.04)
    }
}

// MARK: - Legacy Compatibility

public typealias MLXRAnimation = MLXRMotion

// MARK: - Appear Transition

public struct AppearModifier: ViewModifier {
    let isPresented: Bool

    public func body(content: Content) -> some View {
        content
            .scaleEffect(isPresented ? 1.0 : 0.92)
            .opacity(isPresented ? 1.0 : 0.0)
            .animation(MLXRMotion.spring, value: isPresented)
    }
}

// MARK: - Hover Scale Effect

public struct HoverScaleModifier: ViewModifier {
    @State private var isHovered = false

    let scale: CGFloat
    let shadowRadius: CGFloat

    public init(scale: CGFloat = 1.02, shadowRadius: CGFloat = 12) {
        self.scale = scale
        self.shadowRadius = shadowRadius
    }

    public func body(content: Content) -> some View {
        content
            .scaleEffect(isHovered ? scale : 1.0)
            .shadow(
                color: .black.opacity(isHovered ? 0.15 : 0.04),
                radius: isHovered ? shadowRadius : 4,
                y: isHovered ? 4 : 2
            )
            .animation(MLXRMotion.cardHover, value: isHovered)
            .onHover { hovering in
                isHovered = hovering
            }
    }
}

// MARK: - Staggered Entrance

public struct StaggeredAppear: ViewModifier {
    let index: Int
    @State private var appeared = false

    public func body(content: Content) -> some View {
        content
            .opacity(appeared ? 1.0 : 0.0)
            .offset(y: appeared ? 0 : 12)
            .onAppear {
                withAnimation(MLXRMotion.stagger(index: index)) {
                    appeared = true
                }
            }
    }
}

extension View {
    public func hoverScale(_ scale: CGFloat = 1.02, shadowRadius: CGFloat = 12) -> some View {
        modifier(HoverScaleModifier(scale: scale, shadowRadius: shadowRadius))
    }

    public func staggeredAppear(index: Int) -> some View {
        modifier(StaggeredAppear(index: index))
    }
}

// MARK: - Nav Transitions

@MainActor
public enum MLXRTransition {
    public static let navPush = AnyTransition.asymmetric(
        insertion: .move(edge: .trailing).combined(with: .opacity),
        removal: .move(edge: .leading).combined(with: .opacity)
    )

    public static let navPop = AnyTransition.asymmetric(
        insertion: .move(edge: .leading).combined(with: .opacity),
        removal: .move(edge: .trailing).combined(with: .opacity)
    )
}
