import Foundation
import MLXRAppDomain

enum JSONValueEncoder {
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    static func encode(_ value: JSONMap) throws -> Data {
        let payload = value.mapValues(Self.foundationObject)
        return try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    }

    static func encodeToValue(_ value: some Encodable) throws -> JSONValue {
        let data = try encoder.encode(value)
        let object = try JSONSerialization.jsonObject(with: data)
        return try jsonValue(from: object)
    }

    private static func foundationObject(from value: JSONValue) -> Any {
        switch value {
        case let .string(value):
            value
        case let .number(value):
            value
        case let .integer(value):
            value
        case let .bool(value):
            value
        case let .object(value):
            value.mapValues(foundationObject)
        case let .array(value):
            value.map(foundationObject)
        case .null:
            NSNull()
        }
    }

    private static func jsonValue(from object: Any) throws -> JSONValue {
        switch object {
        case let value as String:
            .string(value)
        case let value as Int:
            .integer(value)
        case let value as Double:
            .number(value)
        case let value as Bool:
            .bool(value)
        case let value as [String: Any]:
            .object(try value.mapValues(jsonValue))
        case let value as [Any]:
            .array(try value.map(jsonValue))
        case is NSNull:
            .null
        default:
            throw RuntimeBridgeError.invalidResponse("Unsupported JSON encoding payload.")
        }
    }
}
