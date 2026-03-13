import Foundation
import MLXRAppDomain

struct EventStreamParser {
    private var bufferedText = ""

    mutating func feed(_ chunk: String) -> [ServerSentEvent] {
        bufferedText.append(chunk)
        var events: [ServerSentEvent] = []

        while let range = bufferedText.range(of: "\n\n") {
            let frame = String(bufferedText[..<range.lowerBound])
            bufferedText.removeSubrange(..<range.upperBound)
            if let event = Self.parseFrame(frame) {
                events.append(event)
            }
        }

        return events
    }

    private static func parseFrame(_ frame: String) -> ServerSentEvent? {
        let normalized = frame.replacingOccurrences(of: "\r\n", with: "\n")
        var eventName: String?
        var dataLines: [String] = []

        for line in normalized.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("event:") {
                eventName = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                dataLines.append(String(line.dropFirst("data:".count)).trimmingCharacters(in: .whitespaces))
            }
        }

        guard let eventName, !dataLines.isEmpty else {
            return nil
        }
        return ServerSentEvent(event: eventName, data: dataLines.joined(separator: "\n"))
    }
}

struct ServerSentEvent: Sendable, Hashable {
    var event: String
    var data: String
}
