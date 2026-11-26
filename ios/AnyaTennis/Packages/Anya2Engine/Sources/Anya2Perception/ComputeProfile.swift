// ComputeProfile.swift
//
// Where the pose model runs on THIS device: the fastest hardware Core ML can
// use here, plus the planning hints the OS version offers.
//
//   simulator                     CPU (it has no Neural Engine, and its GPU path
//                                 is not the device's)
//   Neural Engine usable (A12+)   .all: Core ML puts the network on the Neural
//                                 Engine and spills anything it can't run to
//                                 the GPU/CPU
//   no usable Neural Engine       .cpuAndGPU. The iPhone X / 8 (A11) have a
//                                 Neural Engine, but Core ML can't use it, so
//                                 the GPU is their fastest option
//
// How "usable" is decided:
//   iOS 17+ / macOS 14+   MLModel.availableComputeDevices lists .neuralEngine
//   iOS 16                no such API; Metal GPU family apple5 means A12 or
//                         later, the first chips whose Neural Engine Core ML
//                         runs third-party models on
//
// Hints (each pose instance keeps ONE input shape: near 640x384, far fixed
// per match, see Perceiver):
//   iOS 17.4+ / macOS 14.4+   reshapeFrequency = .infrequent: plan once for
//                             that shape and keep the plan
// NOT specializationStrategy = .fastPrediction (iOS 18): anya2-bench on an M4
// measured no steady-state gain (7.53 vs 7.47 ms/sample) for a 20.6 s first
// load (7.7 s without it), and the far shape changes from match to match.
//
// Debug builds can force a choice for A/B timing on a device with the
// UserDefaults key AnyaComputeUnits = cpu | gpu | ane | all.
import CoreML
import Foundation
import Metal

public struct ComputeProfile: CustomStringConvertible {
    public enum Hardware: String {
        case cpu = "CPU"
        case gpu = "GPU"
        case neuralEngine = "Neural Engine"
    }

    /// What the model is expected to run on (Core ML may still put single ops elsewhere).
    public let hardware: Hardware
    public let computeUnits: MLComputeUnits
    /// Why: how the hardware was found, or that it was forced.
    public let reason: String

    /// This device's profile, decided once per process.
    public static let current = ComputeProfile.detect()

    public init(hardware: Hardware, computeUnits: MLComputeUnits, reason: String) {
        self.hardware = hardware
        self.computeUnits = computeUnits
        self.reason = reason
    }

    /// A model configuration for this profile, with the OS's planning hints.
    public func configuration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        Self.applyHints(to: configuration)
        return configuration
    }

    /// The hints this OS supports, for any compute choice.
    static func applyHints(to configuration: MLModelConfiguration) {
        if #available(iOS 17.4, macOS 14.4, *) {
            var hints = configuration.optimizationHints
            hints.reshapeFrequency = .infrequent
            configuration.optimizationHints = hints
        }
    }

    /// The hint names applied on this OS, for the log.
    static var hintSummary: String {
        if #available(iOS 17.4, macOS 14.4, *) { return "reshape .infrequent" }
        return "no planning hints (OS too old)"
    }

    public var description: String {
        "pose on \(hardware.rawValue) (\(reason); \(Self.hintSummary))"
    }

    // ── detection ─────────────────────────────────────────────────────────

    static func detect() -> ComputeProfile {
        #if targetEnvironment(simulator)
        return ComputeProfile(hardware: .cpu, computeUnits: .cpuOnly, reason: "simulator")
        #else
        #if DEBUG
        if let forced = forcedProfile() { return forced }
        #endif
        switch neuralEngineUsable() {
        case (true, let how):
            return ComputeProfile(hardware: .neuralEngine, computeUnits: .all, reason: how)
        case (false, let how):
            if MTLCreateSystemDefaultDevice() != nil {
                return ComputeProfile(hardware: .gpu, computeUnits: .cpuAndGPU, reason: how)
            }
            return ComputeProfile(hardware: .cpu, computeUnits: .cpuOnly, reason: how + ", no Metal GPU")
        }
        #endif
    }

    /// Whether Core ML can run third-party models on this device's Neural Engine, and how that was decided.
    static func neuralEngineUsable() -> (Bool, String) {
        if #available(iOS 17.0, macOS 14.0, *) {
            let listed = MLModel.availableComputeDevices.contains { device in
                if case .neuralEngine = device { return true }
                return false
            }
            return (listed, listed ? "Core ML lists a Neural Engine" : "Core ML lists no Neural Engine")
        }
        // iOS 16: A12 (GPU family apple5) and later.
        guard let gpu = MTLCreateSystemDefaultDevice() else { return (false, "no Metal device") }
        let a12OrLater = gpu.supportsFamily(.apple5)
        return (a12OrLater, a12OrLater ? "A12 or later" : "A11 or earlier: Neural Engine not available to Core ML")
    }

    #if DEBUG
    static func forcedProfile() -> ComputeProfile? {
        switch UserDefaults.standard.string(forKey: "AnyaComputeUnits")?.lowercased() {
        case "cpu": return ComputeProfile(hardware: .cpu, computeUnits: .cpuOnly, reason: "forced")
        case "gpu": return ComputeProfile(hardware: .gpu, computeUnits: .cpuAndGPU, reason: "forced")
        case "ane": return ComputeProfile(hardware: .neuralEngine, computeUnits: .cpuAndNeuralEngine, reason: "forced")
        case "all": return ComputeProfile(hardware: .neuralEngine, computeUnits: .all, reason: "forced")
        default: return nil
        }
    }
    #endif
}
