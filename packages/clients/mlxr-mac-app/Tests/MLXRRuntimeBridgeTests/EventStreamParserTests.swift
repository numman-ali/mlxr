import Testing
@testable import MLXRRuntimeBridge

@Test
func parserReassemblesChunkedServerSentEvents() {
    var parser = EventStreamParser()

    let first = parser.feed("event: job.progress\ndata: {\"job_id\":\"job_1\"")
    #expect(first.isEmpty)

    let second = parser.feed(",\"kind\":\"job.progress\"}\n\n")
    #expect(second.count == 1)
    #expect(second[0].event == "job.progress")
    #expect(second[0].data == "{\"job_id\":\"job_1\",\"kind\":\"job.progress\"}")
}

@Test
func parserIgnoresFramesWithoutEventAndData() {
    var parser = EventStreamParser()

    let events = parser.feed(": keepalive\n\n")

    #expect(events.isEmpty)
}
