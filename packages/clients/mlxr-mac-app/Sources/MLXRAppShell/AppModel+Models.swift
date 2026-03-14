import Foundation
import MLXRAppDomain
import MLXRRuntimeBridge

@MainActor
extension MLXRAppModel {
    private static let activeInstallPollIntervalNanoseconds: UInt64 = 3_000_000_000

    public func queueModelInstall(modelId: String) async {
        guard let runtime else { return }
        do {
            let operation = try await runtime.enqueueModelInstall(modelId: modelId)
            installOperations = upsert(operation, into: installOperations)
            if modelPreviews[modelId] == nil, let preview = operation.preview {
                modelPreviews[modelId] = preview
            }
            modelsError = nil
            syncInstallMonitor()
        } catch {
            modelsError = error.localizedDescription
        }
    }

    public func cancelModelInstall(operationId: String) async {
        guard let runtime else { return }
        do {
            let operation = try await runtime.cancelModelInstall(operationId: operationId)
            installOperations = upsert(operation, into: installOperations)
            modelsError = nil
            syncInstallMonitor()
        } catch {
            modelsError = error.localizedDescription
        }
    }

    public func loadPreviewIfNeeded(modelId: String) async {
        guard modelPreviews[modelId] == nil, let runtime else { return }
        do {
            modelPreviews[modelId] = try await runtime.previewSupportedModel(modelId: modelId)
        } catch {
            modelsError = error.localizedDescription
        }
    }

    public func loadSetupPreviewsIfNeeded() async {
        let candidateIds = catalog.recommendedAvailableItems.map(\.modelId)
        for modelId in candidateIds where modelPreviews[modelId] == nil {
            await loadPreviewIfNeeded(modelId: modelId)
        }
    }

    public func loadInstalledModelDetails(modelId: String) async {
        guard installedModelDetails[modelId] == nil, let runtime else { return }
        do {
            installedModelDetails[modelId] = try await runtime.modelDetails(modelId: modelId)
            modelsError = nil
        } catch {
            modelsError = error.localizedDescription
        }
    }

    public func removeInstalledModel(modelId: String) async {
        guard let runtime else { return }
        do {
            _ = try await runtime.removeModel(modelId: modelId)
            installedModelDetails.removeValue(forKey: modelId)
            modelPreviews.removeValue(forKey: modelId)
            modelsError = nil
            await refreshCatalogOnly(force: true)
        } catch {
            modelsError = error.localizedDescription
        }
    }

    public func managedLocationURL(for modelId: String) -> URL? {
        guard
            let runtimeStatus,
            let storageKey = installedModelDetails[modelId]?.managedStorageKey
        else {
            return nil
        }
        return runtimeStatus.runtimeHome.appending(path: storageKey, directoryHint: .isDirectory)
    }

    public func installOperation(for modelId: String) -> ModelInstallOperationRecord? {
        installOperations.first { $0.modelId == modelId && !$0.phase.isTerminal }
            ?? installOperations.first { $0.modelId == modelId }
    }

    public func completeModelSetup() {
        hasCompletedOnboarding = true
    }

    func refreshInstallOperationsOnly() async {
        guard let runtime else { return }
        do {
            let previousOperations = installOperations
            let updated = try await runtime.listModelInstalls().sorted { $0.updatedAt > $1.updatedAt }
            installOperations = updated
            modelsError = nil
            syncInstallMonitor()
            if shouldRefreshCatalog(previous: previousOperations, current: updated) {
                await refreshCatalogOnly(force: true)
            }
        } catch {
            modelsError = error.localizedDescription
        }
    }

    func syncInstallMonitor() {
        guard runtime is RuntimeClient else {
            installMonitorTask?.cancel()
            installMonitorTask = nil
            return
        }
        let hasActiveInstalls = installOperations.contains(where: { !$0.phase.isTerminal })
        guard hasActiveInstalls else {
            installMonitorTask?.cancel()
            installMonitorTask = nil
            return
        }
        guard installMonitorTask == nil else {
            return
        }

        installMonitorTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: Self.activeInstallPollIntervalNanoseconds)
                guard let self else { return }
                let shouldContinue = await MainActor.run {
                    self.installOperations.contains(where: { !$0.phase.isTerminal })
                }
                guard shouldContinue else { break }
                await self.refreshInstallOperationsOnly()
            }
            await MainActor.run { [weak self] in
                self?.installMonitorTask = nil
            }
        }
    }

    private func shouldRefreshCatalog(
        previous: [ModelInstallOperationRecord],
        current: [ModelInstallOperationRecord]
    ) -> Bool {
        let previousById = Dictionary(uniqueKeysWithValues: previous.map { ($0.operationId, $0.phase) })
        for operation in current {
            let previousPhase = previousById[operation.operationId]
            if previousPhase != operation.phase, operation.phase.isTerminal {
                return true
            }
        }
        return false
    }

    private func upsert(
        _ operation: ModelInstallOperationRecord,
        into existing: [ModelInstallOperationRecord]
    ) -> [ModelInstallOperationRecord] {
        var updated = existing.filter { $0.operationId != operation.operationId }
        updated.insert(operation, at: 0)
        return updated.sorted { $0.updatedAt > $1.updatedAt }
    }
}
