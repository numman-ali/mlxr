import MLXRDesignSystem
import SwiftUI

struct ComposerControlPill: View {
    struct Option: Identifiable, Hashable {
        let value: String
        let label: String

        var id: String { value }
    }

    let title: String
    let options: [Option]
    let action: (Option) -> Void

    var body: some View {
        Menu {
            ForEach(options) { option in
                Button(option.label) {
                    action(option)
                }
            }
        } label: {
            HStack(spacing: MLXRSpacing.xxs) {
                Text(title)
                    .font(MLXRType.bodySmall)
                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .semibold))
            }
            .foregroundStyle(MLXRColor.textPrimary)
            .padding(.horizontal, MLXRSpacing.sm)
            .padding(.vertical, MLXRSpacing.xs)
            .background(
                Capsule(style: .continuous)
                    .fill(MLXRColor.surfaceHover)
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                    )
            )
        }
        .menuStyle(.borderlessButton)
    }
}
