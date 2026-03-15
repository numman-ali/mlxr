import MLXRDesignSystem
import SwiftUI

struct MediaTileSurface<Content: View>: View {
    let isSelected: Bool
    let cornerRadius: CGFloat
    let action: () -> Void
    @ViewBuilder let content: () -> Content

    init(
        isSelected: Bool = false,
        cornerRadius: CGFloat = MLXRRadius.md,
        action: @escaping () -> Void,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.isSelected = isSelected
        self.cornerRadius = cornerRadius
        self.action = action
        self.content = content
    }

    var body: some View {
        Button(action: action) {
            content()
                .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .strokeBorder(
                            isSelected ? MLXRColor.brandPrimary : MLXRColor.borderSubtle.opacity(0.16),
                            lineWidth: isSelected ? 2 : 0.75
                        )
                }
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}
