// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "grpc_client.h"
#include "gtest/gtest.h"

namespace tc = triton::client;

namespace {

constexpr char kServerUrl[] = "localhost:8001";
constexpr char kLocalCancelMessage[] = "Locally cancelled by application!";

// Time we sleep after sending the request before issuing Cancel().
constexpr auto kInflightWait = std::chrono::seconds(2);

// Generous upper bound for callback delivery once Cancel() has been called.
// The cancel itself should arrive much faster than the 10s model delay.
constexpr auto kCallbackTimeout = std::chrono::seconds(5);

// Upper bound for natural (non-cancelled) completion. The model's per-
// request delay is 10s, so this leaves ~5s of headroom for jitter.
constexpr auto kCompletionTimeout = std::chrono::seconds(15);

class GrpcCancellationTest : public ::testing::Test {
 public:
  void SetUp() override
  {
    auto err = tc::InferenceServerGrpcClient::Create(&client_, kServerUrl);
    ASSERT_TRUE(err.IsOk())
        << "failed to create GRPC client: " << err.Message();
  }

  // Blocks up to `timeout` for callback_ to capture the first result.
  // Returns true if a result was received within the timeout, false on
  // timeout.
  bool WaitForResult(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lk(mutex_);
    return cv_.wait_for(lk, timeout, [this] { return result_ != nullptr; });
  }

  // Returns the number of times callback_ has been invoked. Safe to
  // call from the test thread at any time.
  int GetCallbackCount()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return callback_count_;
  }

  // Clears the captured result and counter so a single test can reuse the
  // fixture across multiple phases (e.g., the cancel-then-restart test).
  void ResetCapturedResult()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    result_.reset();
    callback_count_ = 0;
  }

  // Builds a single INPUT0/OUTPUT0 pair matching the Python suite's
  // _prepare_request(). Caller owns the returned pointers.
  tc::Error PrepareRequest(
      std::vector<tc::InferInput*>* inputs,
      std::vector<const tc::InferRequestedOutput*>* outputs)
  {
    inputs->emplace_back();
    auto err =
        tc::InferInput::Create(&inputs->back(), "INPUT0", shape_, dtype_);
    if (!err.IsOk()) {
      return err;
    }
    err = inputs->back()->AppendRaw(
        reinterpret_cast<const uint8_t*>(input_data_.data()),
        input_data_.size() * sizeof(int32_t));
    if (!err.IsOk()) {
      return err;
    }
    tc::InferRequestedOutput* out0 = nullptr;
    err = tc::InferRequestedOutput::Create(&out0, "OUTPUT0");
    if (!err.IsOk()) {
      return err;
    }
    outputs->push_back(out0);
    return tc::Error::Success;
  }

  static void CleanupRequest(
      std::vector<tc::InferInput*>* inputs,
      std::vector<const tc::InferRequestedOutput*>* outputs)
  {
    for (auto* in : *inputs) {
      delete in;
    }
    for (auto* out : *outputs) {
      delete const_cast<tc::InferRequestedOutput*>(out);
    }
    inputs->clear();
    outputs->clear();
  }

  tc::InferOptions MakeOptions() const
  {
    tc::InferOptions options(model_name_);
    options.model_version_ = model_version_;
    return options;
  }

  // Verifies a successful response echoes input_data_ on OUTPUT0 with
  // matching shape and dtype. Run on the test thread after the callback
  // has stashed the result.
  void ExpectEchoOutput(tc::InferResult* result) const
  {
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->RequestStatus().IsOk())
        << "request failed: " << result->RequestStatus().Message();

    std::vector<int64_t> out_shape;
    EXPECT_TRUE(result->Shape("OUTPUT0", &out_shape).IsOk());
    EXPECT_EQ(out_shape, shape_);

    std::string out_dtype;
    EXPECT_TRUE(result->Datatype("OUTPUT0", &out_dtype).IsOk());
    EXPECT_EQ(out_dtype, dtype_);

    const uint8_t* buf = nullptr;
    size_t byte_size = 0;
    ASSERT_TRUE(result->RawData("OUTPUT0", &buf, &byte_size).IsOk());
    const size_t expected_bytes = input_data_.size() * sizeof(int32_t);
    ASSERT_EQ(byte_size, expected_bytes);
    ASSERT_NE(buf, nullptr);
    std::vector<int32_t> out_data(input_data_.size());
    std::memcpy(out_data.data(), buf, byte_size);
    EXPECT_EQ(out_data, input_data_);
  }

  static void ExpectLocalCancel(tc::InferResult* result)
  {
    ASSERT_NE(result, nullptr);
    const tc::Error st = result->RequestStatus();
    EXPECT_FALSE(st.IsOk())
        << "expected cancellation, but request reported success";
    EXPECT_NE(st.Message().find(kLocalCancelMessage), std::string::npos)
        << "expected '" << kLocalCancelMessage
        << "' in status, got: " << st.Message();
  }

  const std::string model_name_ = "custom_identity_int32";
  const std::string model_version_ = "1";
  const std::vector<int32_t> input_data_ = {10};
  const std::vector<int64_t> shape_ = {1, 1};
  const std::string dtype_ = "INT32";
  std::unique_ptr<tc::InferenceServerGrpcClient> client_;

  // Synchronization for callbacks fired on the gRPC worker thread. Tests
  // pass callback_ into AsyncInfer/StartStream and then either call
  // WaitForResult() or sleep before inspecting result_/callback_count_.
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unique_ptr<tc::InferResult> result_;
  int callback_count_ = 0;

  // Stashes the first InferResult into result_ and counts every invocation.
  // Subsequent results are deleted so a buggy double-callback doesn't leak;
  // tests then catch it by checking GetCallbackCount().
  tc::InferenceServerClient::OnCompleteFn callback_ =
      [this](tc::InferResult* r) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (result_ == nullptr) {
          result_.reset(r);
        } else {
          delete r;
        }
        ++callback_count_;
        cv_.notify_one();
      };
};

// ---------------------------------------------------------------------------
// Unary cancellation tests.
// ---------------------------------------------------------------------------

// Launch an async inference, sleep briefly so the RPC is in flight, call
// Cancel() on the returned CallContext, and assert the callback received a
// CANCELLED status the cancellation message.
TEST_F(GrpcCancellationTest, TestGrpcAsyncInfer)
{
  std::vector<tc::InferInput*> inputs;
  std::vector<const tc::InferRequestedOutput*> outputs;
  ASSERT_TRUE(PrepareRequest(&inputs, &outputs).IsOk());

  tc::CallContext* raw_ctx = nullptr;
  ASSERT_TRUE(client_
                  ->AsyncInfer(
                      callback_, MakeOptions(), inputs, outputs,
                      /*headers=*/{}, GRPC_COMPRESS_NONE, &raw_ctx)
                  .IsOk());
  ASSERT_NE(raw_ctx, nullptr);
  std::unique_ptr<tc::CallContext> ctx(raw_ctx);

  std::this_thread::sleep_for(kInflightWait);
  ASSERT_TRUE(ctx->Cancel().IsOk());

  ASSERT_TRUE(WaitForResult(kCallbackTimeout))
      << "callback was never invoked after Cancel()";
  ExpectLocalCancel(result_.get());

  CleanupRequest(&inputs, &outputs);
}

// C++-specific lifetime check: the CallContext co-owns the ClientContext
// and may outlive the RPC. Cancel() after natural completion must be a safe
// no-op and must not double-fire the callback.
TEST_F(GrpcCancellationTest, TestGrpcAsyncInferCancelAfterCompletionIsNoOp)
{
  std::vector<tc::InferInput*> inputs;
  std::vector<const tc::InferRequestedOutput*> outputs;
  ASSERT_TRUE(PrepareRequest(&inputs, &outputs).IsOk());

  tc::CallContext* raw_ctx = nullptr;
  ASSERT_TRUE(client_
                  ->AsyncInfer(
                      callback_, MakeOptions(), inputs, outputs,
                      /*headers=*/{}, GRPC_COMPRESS_NONE, &raw_ctx)
                  .IsOk());
  ASSERT_NE(raw_ctx, nullptr);
  std::unique_ptr<tc::CallContext> ctx(raw_ctx);

  ASSERT_TRUE(WaitForResult(kCompletionTimeout)) << "inference never completed";
  ExpectEchoOutput(result_.get());

  ASSERT_TRUE(ctx->Cancel().IsOk());
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  ASSERT_EQ(GetCallbackCount(), 1)
      << "Cancel() after completion produced an extra callback";

  CleanupRequest(&inputs, &outputs);
}

// Backwards-compat: AsyncInfer() called without ctx_out (the default arg)
// keeps its previous behavior and does not return a handle.
TEST_F(GrpcCancellationTest, TestGrpcAsyncInferWithoutContextStillCompletes)
{
  std::vector<tc::InferInput*> inputs;
  std::vector<const tc::InferRequestedOutput*> outputs;
  ASSERT_TRUE(PrepareRequest(&inputs, &outputs).IsOk());

  ASSERT_TRUE(
      client_->AsyncInfer(callback_, MakeOptions(), inputs, outputs).IsOk());

  ASSERT_TRUE(WaitForResult(kCompletionTimeout))
      << "inference never completed without CallContext";
  ExpectEchoOutput(result_.get());

  CleanupRequest(&inputs, &outputs);
}

// ---------------------------------------------------------------------------
// AsyncInferMulti cancellation tests.
// ---------------------------------------------------------------------------

// C++-specific: AsyncInferMulti fans one logical batch out to N AsyncInfer
// calls and fires its OnMultiCompleteFn once after all leaves complete.
// Cancel requests 0 and 2 while letting request 1 complete naturally to
// verify cancellation is per-request (siblings unaffected), result order
// matches input order, and the multi callback fires exactly once.
TEST_F(GrpcCancellationTest, TestGrpcAsyncInferMulti)
{
  constexpr size_t kNumRequests = 3;
  constexpr size_t kSucceedIndex = 1;

  std::vector<std::vector<tc::InferInput*>> inputs(kNumRequests);
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs(
      kNumRequests);
  for (size_t i = 0; i < kNumRequests; ++i) {
    ASSERT_TRUE(PrepareRequest(&inputs[i], &outputs[i]).IsOk());
  }

  // OnMultiCompleteFn signature differs from the fixture's callback_,
  // so synchronize locally.
  std::mutex local_mutex;
  std::condition_variable local_cv;
  std::vector<tc::InferResult*> captured_results;
  int multi_callback_count = 0;
  auto multi_callback =
      [&local_mutex, &local_cv, &captured_results,
       &multi_callback_count](std::vector<tc::InferResult*> results) {
        std::lock_guard<std::mutex> lk(local_mutex);
        captured_results = std::move(results);
        ++multi_callback_count;
        local_cv.notify_one();
      };

  std::vector<tc::CallContext*> raw_ctxs;
  std::vector<tc::InferOptions> options = {MakeOptions()};
  ASSERT_TRUE(client_
                  ->AsyncInferMulti(
                      multi_callback, options, inputs, outputs,
                      /*headers=*/{}, GRPC_COMPRESS_NONE, &raw_ctxs)
                  .IsOk());
  ASSERT_EQ(raw_ctxs.size(), kNumRequests);

  // Own the returned contexts so they get deleted even if an ASSERT fires.
  std::vector<std::unique_ptr<tc::CallContext>> ctxs;
  ctxs.reserve(raw_ctxs.size());
  for (auto* raw : raw_ctxs) {
    ASSERT_NE(raw, nullptr);
    ctxs.emplace_back(raw);
  }

  std::this_thread::sleep_for(kInflightWait);
  for (size_t i = 0; i < ctxs.size(); ++i) {
    if (i != kSucceedIndex) {
      ASSERT_TRUE(ctxs[i]->Cancel().IsOk());
    }
  }

  {
    // Multi callback fires only after every leaf completes, so the
    // uncancelled request must run the full model delay; use the
    // completion-sized timeout, not the cancel-sized one.
    std::unique_lock<std::mutex> lk(local_mutex);
    ASSERT_TRUE(local_cv.wait_for(lk, kCompletionTimeout, [&] {
      return multi_callback_count > 0;
    })) << "multi callback was never invoked";
    EXPECT_EQ(multi_callback_count, 1) << "multi callback fired more than once";
    ASSERT_EQ(captured_results.size(), kNumRequests);
    for (size_t i = 0; i < captured_results.size(); ++i) {
      if (i == kSucceedIndex) {
        ExpectEchoOutput(captured_results[i]);
      } else {
        ExpectLocalCancel(captured_results[i]);
      }
    }
  }

  for (auto* r : captured_results) {
    delete r;
  }
  for (size_t i = 0; i < kNumRequests; ++i) {
    CleanupRequest(&inputs[i], &outputs[i]);
  }
}

// ---------------------------------------------------------------------------
// Streaming cancellation tests.
// ---------------------------------------------------------------------------

// Start a stream, send 3 inferences, sleep so the requests are in flight on
// the server, then call StopStream(cancel_requests=true) and assert the
// callback received the cancellation message.
TEST_F(GrpcCancellationTest, TestGrpcStreamInfer)
{
  constexpr int kNumStreamRequests = 3;

  ASSERT_TRUE(client_->StartStream(callback_, /*enable_stats=*/false).IsOk());

  std::vector<std::vector<tc::InferInput*>> inputs(kNumStreamRequests);
  std::vector<std::vector<const tc::InferRequestedOutput*>> outputs(
      kNumStreamRequests);
  for (int i = 0; i < kNumStreamRequests; ++i) {
    ASSERT_TRUE(PrepareRequest(&inputs[i], &outputs[i]).IsOk());
    ASSERT_TRUE(
        client_->AsyncStreamInfer(MakeOptions(), inputs[i], outputs[i]).IsOk());
  }

  std::this_thread::sleep_for(kInflightWait);
  ASSERT_TRUE(client_->StopStream(/*cancel_requests=*/true).IsOk());

  ASSERT_TRUE(WaitForResult(kCallbackTimeout))
      << "stream callback was never invoked after StopStream(cancel)";
  ExpectLocalCancel(result_.get());
  EXPECT_EQ(GetCallbackCount(), 1)
      << "expected exactly one synthesized cancel callback per stream, got "
      << GetCallbackCount();

  for (int i = 0; i < kNumStreamRequests; ++i) {
    CleanupRequest(&inputs[i], &outputs[i]);
  }
}

// C++-specific safety check: cancelling a freshly-started stream that has
// never sent a request should still surface cancellation message via the
// callback exactly once.
TEST_F(GrpcCancellationTest, TestGrpcStreamCancelWithoutInfer)
{
  ASSERT_TRUE(client_->StartStream(callback_, /*enable_stats=*/false).IsOk());
  ASSERT_TRUE(client_->StopStream(/*cancel_requests=*/true).IsOk());

  ASSERT_TRUE(WaitForResult(kCallbackTimeout))
      << "stream callback was never invoked after StopStream(cancel)";
  ExpectLocalCancel(result_.get());
}

// C++-specific lifecycle check: after a cancelled stream, the client must
// be able to start a fresh stream and run a successful inference. Mirrors
// the rebind-grpc_context_ behavior added in StartStream().
TEST_F(GrpcCancellationTest, TestGrpcStreamCancelThenRestart)
{
  // ---- Phase 1: start a stream and cancel it before sending any request.
  ASSERT_TRUE(client_->StartStream(callback_, /*enable_stats=*/false).IsOk());
  ASSERT_TRUE(client_->StopStream(/*cancel_requests=*/true).IsOk());
  ASSERT_TRUE(WaitForResult(kCallbackTimeout))
      << "stream callback was never invoked after StopStream(cancel)";
  ExpectLocalCancel(result_.get());

  // ---- Phase 2: start a fresh stream and run a successful inference.
  // Clear fixture state so WaitForResult() observes the new callback, not
  // the leftover phase-1 result.
  ResetCapturedResult();

  ASSERT_TRUE(client_->StartStream(callback_, /*enable_stats=*/false).IsOk());

  std::vector<tc::InferInput*> inputs;
  std::vector<const tc::InferRequestedOutput*> outputs;
  ASSERT_TRUE(PrepareRequest(&inputs, &outputs).IsOk());
  ASSERT_TRUE(client_->AsyncStreamInfer(MakeOptions(), inputs, outputs).IsOk());

  // Graceful close: WritesDone + drain. Blocks until the slow model's
  // response arrives, so the success callback has fired by the time the
  // call below returns.
  ASSERT_TRUE(client_->StopStream(/*cancel_requests=*/false).IsOk());

  ASSERT_TRUE(WaitForResult(kCompletionTimeout))
      << "expected successful inference after restarting stream";
  ExpectEchoOutput(result_.get());

  CleanupRequest(&inputs, &outputs);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
