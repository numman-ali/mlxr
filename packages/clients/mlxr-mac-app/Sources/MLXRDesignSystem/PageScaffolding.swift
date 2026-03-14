import SwiftUI

public struct CompactPageHeader<Content: View>: View {
    private let title: String
    private let subtitle: String?
    private let content: Content

    public init(
        title: String,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.sm) {
            HStack(alignment: .top, spacing: MLXRSpacing.lg) {
                VStack(alignment: .leading, spacing: MLXRSpacing.xxs) {
                    Text(title)
                        .font(MLXRType.titleLarge)
                        .foregroundStyle(MLXRColor.textPrimary)

                    if let subtitle, !subtitle.isEmpty {
                        Text(subtitle)
                            .font(MLXRType.bodySmall)
                            .foregroundStyle(MLXRColor.textSecondary)
                            .lineLimit(2)
                    }
                }

                Spacer(minLength: 0)

                content
            }

            Divider()
        }
    }
}

public struct DenseSectionSurface<Content: View>: View {
    private let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: MLXRSpacing.md) {
            content
        }
        .padding(MLXRSpacing.lg)
        .background(
            RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                .fill(MLXRColor.surfaceCard)
                .overlay(
                    RoundedRectangle(cornerRadius: MLXRRadius.lg, style: .continuous)
                        .strokeBorder(MLXRColor.borderSubtle, lineWidth: 0.5)
                )
        )
    }
}
