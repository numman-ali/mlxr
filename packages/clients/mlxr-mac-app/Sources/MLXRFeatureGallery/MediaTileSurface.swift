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
                .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .strokeBorder(
                            isSelected ? MLXRColor.brandPrimary : MLXRColor.surfaceCard,
                            lineWidth: isSelected ? 2 : 1
                        )
                }
                .background(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(MLXRColor.canvasRaised)
                )
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}
