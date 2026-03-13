import Foundation
import MLXRAppDomain
import NIOCore
import NIOHTTP1
import NIOPosix

struct UnixDomainSocketHTTPResponse: Sendable {
    let status: HTTPResponseStatus
    let headers: HTTPHeaders
    let body: Data
}

final class UnixDomainSocketHTTPClient: @unchecked Sendable {
    private let group: MultiThreadedEventLoopGroup

    init(group: MultiThreadedEventLoopGroup) {
        self.group = group
    }

    func execute(
        method: HTTPMethod,
        socketPath: String,
        uri: String,
        headers: HTTPHeaders,
        body: Data?
    ) async throws -> UnixDomainSocketHTTPResponse {
        try await withCheckedThrowingContinuation { continuation in
            let completion = ContinuationBox(continuation)
            let handler = OneShotUnixRequestHandler(
                method: method,
                uri: uri,
                headers: self.preparedHeaders(headers, bodyLength: body?.count),
                body: body,
                completion: completion
            )
            let bootstrap = self.makeBootstrap(handler: handler)

            Task {
                do {
                    _ = try await bootstrap.connect(unixDomainSocketPath: socketPath).get()
                } catch {
                    completion.fail(error)
                }
            }
        }
    }

    func streamEvents(
        socketPath: String,
        uri: String,
        headers: HTTPHeaders,
        decoder: JSONDecoder,
        continuation: AsyncThrowingStream<RuntimeEvent, Error>.Continuation
    ) async throws {
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
            let completion = ContinuationBox(done)
            let handler = EventStreamUnixRequestHandler(
                uri: uri,
                headers: self.preparedHeaders(headers, bodyLength: nil),
                decoder: decoder,
                continuation: continuation,
                completion: completion
            )
            let bootstrap = self.makeBootstrap(handler: handler)

            Task {
                do {
                    _ = try await bootstrap.connect(unixDomainSocketPath: socketPath).get()
                } catch {
                    completion.fail(error)
                }
            }
        }
    }

    private func makeBootstrap<Handler: ChannelHandler & Sendable>(handler: Handler) -> ClientBootstrap {
        ClientBootstrap(group: group)
            .channelOption(.socketOption(.so_reuseaddr), value: 1)
            .channelInitializer { channel in
                channel.eventLoop.makeCompletedFuture {
                    try channel.pipeline.syncOperations.addHTTPClientHandlers(
                        position: .first,
                        leftOverBytesStrategy: .fireError
                    )
                    try channel.pipeline.syncOperations.addHandler(handler)
                }
            }
    }

    private func preparedHeaders(_ headers: HTTPHeaders, bodyLength: Int?) -> HTTPHeaders {
        var prepared = headers
        if prepared.first(name: "host") == nil {
            prepared.add(name: "host", value: "localhost")
        }
        if let bodyLength, prepared.first(name: "content-length") == nil {
            prepared.add(name: "content-length", value: "\(bodyLength)")
        }
        return prepared
    }
}

private final class ContinuationBox<Response: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Response, Error>?

    init(_ continuation: CheckedContinuation<Response, Error>) {
        self.continuation = continuation
    }

    func succeed(_ value: Response) {
        let continuation = take()
        continuation?.resume(returning: value)
    }

    func fail(_ error: Error) {
        let continuation = take()
        continuation?.resume(throwing: error)
    }

    private func take() -> CheckedContinuation<Response, Error>? {
        lock.lock()
        defer { lock.unlock() }
        let continuation = continuation
        self.continuation = nil
        return continuation
    }
}

private final class OneShotUnixRequestHandler: ChannelInboundHandler, ChannelOutboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPClientResponsePart
    typealias OutboundIn = HTTPClientRequestPart
    typealias OutboundOut = HTTPClientRequestPart

    private let method: HTTPMethod
    private let uri: String
    private let headers: HTTPHeaders
    private let body: Data?
    private let completion: ContinuationBox<UnixDomainSocketHTTPResponse>

    private var responseHead: HTTPResponseHead?
    private var responseBody = ByteBufferAllocator().buffer(capacity: 0)

    init(
        method: HTTPMethod,
        uri: String,
        headers: HTTPHeaders,
        body: Data?,
        completion: ContinuationBox<UnixDomainSocketHTTPResponse>
    ) {
        self.method = method
        self.uri = uri
        self.headers = headers
        self.body = body
        self.completion = completion
    }

    func channelActive(context: ChannelHandlerContext) {
        var requestHead = HTTPRequestHead(
            version: .http1_1,
            method: method,
            uri: uri,
            headers: headers
        )
        requestHead.headers = headers

        context.write(Self.wrapOutboundOut(.head(requestHead)), promise: nil)
        if let body {
            var buffer = context.channel.allocator.buffer(capacity: body.count)
            buffer.writeBytes(body)
            context.write(Self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
        context.writeAndFlush(Self.wrapOutboundOut(.end(nil)), promise: nil)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch Self.unwrapInboundIn(data) {
        case let .head(head):
            self.responseHead = head
        case let .body(buffer):
            var buffer = buffer
            self.responseBody.writeBuffer(&buffer)
        case .end:
            guard let head = self.responseHead else {
                self.completion.fail(RuntimeBridgeError.invalidResponse("No HTTP response head received from the runtime."))
                context.close(promise: nil)
                return
            }
            let payload = Data(self.responseBody.readableBytesView)
            self.completion.succeed(
                UnixDomainSocketHTTPResponse(
                    status: head.status,
                    headers: head.headers,
                    body: payload
                )
            )
            context.close(promise: nil)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        if self.responseHead == nil {
            self.completion.fail(RuntimeBridgeError.runtimeUnavailable("The runtime closed the Unix socket before sending a response."))
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        self.completion.fail(error)
        context.close(promise: nil)
    }
}

private final class EventStreamUnixRequestHandler: ChannelInboundHandler, ChannelOutboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPClientResponsePart
    typealias OutboundIn = HTTPClientRequestPart
    typealias OutboundOut = HTTPClientRequestPart

    private let decoder: JSONDecoder
    private let continuation: AsyncThrowingStream<RuntimeEvent, Error>.Continuation
    private let completion: ContinuationBox<Void>
    private let headers: HTTPHeaders
    private let uri: String

    private var statusCode: Int?
    private var parser = EventStreamParser()
    private var errorBuffer = ByteBufferAllocator().buffer(capacity: 0)

    init(
        uri: String,
        headers: HTTPHeaders,
        decoder: JSONDecoder,
        continuation: AsyncThrowingStream<RuntimeEvent, Error>.Continuation,
        completion: ContinuationBox<Void>
    ) {
        self.uri = uri
        self.headers = headers
        self.decoder = decoder
        self.continuation = continuation
        self.completion = completion
    }

    func channelActive(context: ChannelHandlerContext) {
        var requestHead = HTTPRequestHead(
            version: .http1_1,
            method: .GET,
            uri: uri,
            headers: headers
        )
        requestHead.headers = headers
        context.write(Self.wrapOutboundOut(.head(requestHead)), promise: nil)
        context.writeAndFlush(Self.wrapOutboundOut(.end(nil)), promise: nil)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch Self.unwrapInboundIn(data) {
        case let .head(head):
            self.statusCode = Int(head.status.code)
        case let .body(buffer):
            var buffer = buffer
            if let statusCode, statusCode >= 400 {
                self.errorBuffer.writeBuffer(&buffer)
                return
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
                    self.completion.fail(error)
                    context.close(promise: nil)
                    return
                }
            }
        case .end:
            if let statusCode, statusCode >= 400 {
                let detail = String(decoding: self.errorBuffer.readableBytesView, as: UTF8.self)
                self.completion.fail(RuntimeBridgeError.api(statusCode: statusCode, detail: detail))
            } else {
                self.completion.succeed(())
            }
            context.close(promise: nil)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        if self.statusCode == nil {
            self.completion.fail(RuntimeBridgeError.runtimeUnavailable("The runtime closed the Unix socket before opening the event stream."))
        }
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        self.completion.fail(error)
        context.close(promise: nil)
    }
}
