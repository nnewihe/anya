// Theme.swift
//
// The app's look in one place. The colours are the ones the app has always
// used -- dark background, system blue for actions, red for recording, green
// for done, orange for attention, yellow for the court -- set in cards,
// capsule buttons and glass panels over the camera.
//
// iOS 16 APIs only (the app runs on the iPhone X).
import SwiftUI

enum Theme {
    static let background = Color.black
    static let card = Color(white: 0.11)
    static let cardRaised = Color(white: 0.15)
    static let stroke = Color.white.opacity(0.08)
    static let field = Color.white.opacity(0.07)
    static let accent = Color.accentColor
    static let record = Color.red
    static let done = Color.green
    static let warning = Color.orange
    static let court = Color.yellow

    static let radius: CGFloat = 22
    static let smallRadius: CGFloat = 14

    /// Rounded system type, for headings and numbers.
    static func display(_ size: CGFloat, _ weight: Font.Weight = .bold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }
}

// ── surfaces ──────────────────────────────────────────────────────────────

extension View {
    /// A dark rounded card.
    func card(padding: CGFloat = 18, fill: Color = Theme.card) -> some View {
        self.padding(padding)
            .background(fill, in: RoundedRectangle(cornerRadius: Theme.radius, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.radius, style: .continuous).stroke(Theme.stroke))
    }

    /// A glass panel for controls over the camera picture.
    func glass<S: Shape>(_ shape: S) -> some View {
        self.background(.ultraThinMaterial, in: shape)
            .overlay(shape.stroke(Color.white.opacity(0.12)))
            .environment(\.colorScheme, .dark)
    }
}

/// Small caps label above a section.
struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text.uppercased())
            .font(.caption.weight(.semibold))
            .tracking(1.6)
            .foregroundStyle(.secondary)
    }
}

// ── buttons ───────────────────────────────────────────────────────────────

/// Filled capsule: the one main action on a screen.
struct PrimaryButtonStyle: ButtonStyle {
    var tint: Color = Theme.accent
    var large = false
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(large ? .title3.weight(.semibold) : .body.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, large ? 26 : 18)
            .padding(.vertical, large ? 15 : 11)
            .background(tint.opacity(isEnabled ? 1 : 0.35), in: Capsule())
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .opacity(configuration.isPressed ? 0.85 : 1)
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// Quiet capsule: the other actions.
struct SecondaryButtonStyle: ButtonStyle {
    var tint: Color = .white
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.body.weight(.medium))
            .foregroundStyle(tint.opacity(isEnabled ? 1 : 0.4))
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.white.opacity(configuration.isPressed ? 0.16 : 0.09), in: Capsule())
            .animation(.easeOut(duration: 0.15), value: configuration.isPressed)
    }
}

/// A full-width row in an action list (Save, Share, Delete…).
struct ActionRowStyle: ButtonStyle {
    var tint: Color = Theme.accent

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.body.weight(.medium))
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(Color.white.opacity(configuration.isPressed ? 0.12 : 0.05),
                        in: RoundedRectangle(cornerRadius: Theme.smallRadius, style: .continuous))
    }
}

extension ButtonStyle where Self == PrimaryButtonStyle {
    static var primary: PrimaryButtonStyle { PrimaryButtonStyle() }
    static func primary(_ tint: Color, large: Bool = false) -> PrimaryButtonStyle { PrimaryButtonStyle(tint: tint, large: large) }
}

extension ButtonStyle where Self == SecondaryButtonStyle {
    static var secondary: SecondaryButtonStyle { SecondaryButtonStyle() }
}

extension ButtonStyle where Self == ActionRowStyle {
    static var actionRow: ActionRowStyle { ActionRowStyle() }
    static func actionRow(_ tint: Color) -> ActionRowStyle { ActionRowStyle(tint: tint) }
}

/// A round icon button, for Back / Close over any background.
struct CircleIconButton: View {
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.body.weight(.semibold))
                .frame(width: 40, height: 40)
                .glass(Circle())
        }
        .buttonStyle(.plain)
    }
}

// ── small pieces ──────────────────────────────────────────────────────────

/// A number with a label: "3 points", "0:00:37 reel".
struct StatChip: View {
    let value: String
    let label: String
    let systemImage: String
    var tint: Color = Theme.accent

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.callout.weight(.semibold))
                .foregroundStyle(tint)
                .frame(width: 30, height: 30)
                .background(tint.opacity(0.15), in: Circle())
            VStack(alignment: .leading, spacing: 1) {
                Text(value).font(Theme.display(17, .semibold)).monospacedDigit()
                Text(label).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// A match's calibration frame (the court as the camera saw it), or a placeholder.
struct MatchThumbnail: View {
    let matchID: String
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(white: 0.18), Color(white: 0.09)], startPoint: .top, endPoint: .bottom)
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "figure.tennis").font(.system(size: 30)).foregroundStyle(.tertiary)
            }
        }
        .clipped()
        .task(id: matchID) {
            let url = MatchStore.file("calibration.png", of: matchID)
            image = await Task.detached(priority: .utility) { UIImage(contentsOfFile: url.path) }.value
        }
    }
}

/// The app mark and name.
struct Wordmark: View {
    var size: CGFloat = 34

    var body: some View {
        HStack(spacing: 10) {
            Image("Logo").resizable().frame(width: size, height: size)
                .clipShape(RoundedRectangle(cornerRadius: size * 0.24, style: .continuous))
            HStack(spacing: 6) {
                Text("ANYA").font(Theme.display(size * 0.5, .heavy))
                Text("TENNIS").font(Theme.display(size * 0.5, .medium)).foregroundStyle(.secondary)
            }
            .tracking(2)
        }
    }
}

/// A pulsing dot, for "recording now".
struct PulsingDot: View {
    var color: Color = Theme.record
    @State private var on = false

    var body: some View {
        Circle().fill(color).frame(width: 10, height: 10)
            .opacity(on ? 1 : 0.35)
            .onAppear { withAnimation(.easeInOut(duration: 0.8).repeatForever()) { on = true } }
    }
}
