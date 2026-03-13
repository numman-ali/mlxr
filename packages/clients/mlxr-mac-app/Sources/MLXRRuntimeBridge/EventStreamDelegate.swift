import AsyncHTTPClient
import Foundation
import MLXRAppDomain
import NIOCore
import NIOHTTP1

final class EventStreamDelegate: HTTPClientResponseDelegate, @unchecked Sendable {
    typealias Response = Void

    private let decoder: JSONDecoder
    private let errorBuilder: @Sendable (Int, Data) -> RuntimeBridgeError
    private let continuation: AsyncThrowingStream<RuntimeEvent, Error>.Continuation
    private var parser = EventStreamParser()
    private var statusCode: Int?
    private var errorBuffer = ByteBufferAllocator().buffer(capacity: 0)

    init(
        decoder: JSONDecoder,
        errorBuilder: @escaping @Sendable (Int, Data) -> RuntimeBridgeError,
        continuation: AsyncThrowingStream<RuntimeEvent, Error>.Continuation
    ) {
        self.decoder = decoder
        self.errorBuilder = errorBuilder
        self.continuation = continuation
    }

    func didReceiveHead(
        task: HTTPClient.Task<Void>,
        _ head: HTTPResponseHead
    ) -> EventLoopFuture<Void> {
        self.statusCode = Int(head.status.code)
        return task.eventLoop.makeSucceededVoidFuture()
    }

    func didReceiveBodyPart(
        task: HTTPClient.Task<Void>,
        _ buffer: ByteBuffer
    ) -> EventLoopFuture<Void> {
        var buffer = buffer
        if let statusCode = self.statusCode, statusCode >= 400 {
            self.errorBuffer.writeBuffer(&buffer)
            return task.eventLoop.makeSucceededVoidFuture()
        }

        let text = buffer.readString(length: buffer.readableBytes) ?? ""
        for event in parser.feed(text) {
            guard let payload = event.data.data(using: .utf8) else {
                continue
            }
            do {
                let runtimeEvent = try decoder.decode(RuntimeEvent.self, from: payload)
                continuation.yield(runtimeEvent)
            } catch {
                task.fail(reason: error)
                return task.eventLoop.makeFailedFuture(error)
            }
        }
        return task.eventLoop.makeSucceededVoidFuture()
    }

    func didReceiveError(task: HTTPClient.Task<Void>, _ error: Error) {}

    func didFinishRequest(task: HTTPClient.Task<Void>) throws -> Void {
        if let statusCode = self.statusCode, statusCode >= 400 {
            let payload = self.errorBuffer.readData(length: self.errorBuffer.readableBytes) ?? Data()
            throw errorBuilder(statusCode, payload)
        }
    }
}
