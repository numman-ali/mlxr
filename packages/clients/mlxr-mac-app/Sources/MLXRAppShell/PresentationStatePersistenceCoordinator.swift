import Foundation
import MLXRAppDomain

actor PresentationStatePersistenceCoordinator {
    private let store: WorkspaceStateStore
    private var generation = 0

    init(store: WorkspaceStateStore) {
        self.store = store
    }

    func persist(
        _ state: AppPresentationState,
        debounce: Duration = .milliseconds(150)
    ) async throws {
        generation += 1
        let currentGeneration = generation
        try await Task.sleep(for: debounce)
        guard currentGeneration == generation else {
            return
        }
        try store.persist(state)
    }
}
