// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "command_line_parser.h"
#include "doctest.h"
#include "load_manager.h"
#include "test_load_manager_base.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

namespace {

bool
operator==(const RequestRecord& lhs, const RequestRecord& rhs)
{
  return std::tie(
             lhs.start_time_, lhs.response_times_, lhs.sequence_end_,
             lhs.delayed_, lhs.sequence_id_, lhs.has_null_last_response_) ==
         std::tie(
             rhs.start_time_, rhs.response_times_, rhs.sequence_end_,
             rhs.delayed_, rhs.sequence_id_, rhs.has_null_last_response_);
}

}  // namespace

class TestLoadManager : public TestLoadManagerBase, public LoadManager {
 public:
  ~TestLoadManager() = default;
  TestLoadManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false)
      : TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        LoadManager(
            params.async, params.streaming, params.batch_size,
            params.max_threads, params.shared_memory_type,
            params.output_shm_size, GetParser(), GetFactory(),
            params.request_parameters)
  {
  }

  std::vector<std::shared_ptr<ThreadStat>>& threads_stat_{
      LoadManager::threads_stat_};

  /// Test the public function CheckHealth
  ///
  /// It will return a bad result if any of the thread stats
  /// have a bad status or cb_status
  ///
  void TestCheckHealth()
  {
    auto good = std::make_shared<ThreadStat>();
    good->status_ = cb::Error::Success;
    good->cb_status_ = cb::Error::Success;

    auto bad_status = std::make_shared<ThreadStat>();
    bad_status->status_ = cb::Error::Failure;
    bad_status->cb_status_ = cb::Error::Success;

    auto bad_cb_status = std::make_shared<ThreadStat>();
    bad_cb_status->status_ = cb::Error::Success;
    bad_cb_status->cb_status_ = cb::Error::Failure;

    threads_stat_.clear();
    bool expect_ok = true;

    SUBCASE("Empty")
    {
      expect_ok = true;
    }
    SUBCASE("Good")
    {
      // Good entries: expect OK
      threads_stat_.push_back(good);
      threads_stat_.push_back(good);
      expect_ok = true;
    }
    SUBCASE("BadStatus")
    {
      // Bad Status: expect not OK
      threads_stat_.push_back(good);
      threads_stat_.push_back(bad_status);
      expect_ok = false;
    }
    SUBCASE("BadCbStatus")
    {
      // Bad cb_Status: expect not OK
      threads_stat_.push_back(bad_cb_status);
      threads_stat_.push_back(good);
      expect_ok = false;
    }
    SUBCASE("BadBothStatus")
    {
      threads_stat_.push_back(bad_status);
      threads_stat_.push_back(good);
      threads_stat_.push_back(bad_cb_status);
      expect_ok = false;
    }

    CHECK(CheckHealth().IsOk() == expect_ok);
  }

  /// Test the public function SwapRequestRecords
  ///
  /// It will gather all request records from the thread_stats
  /// and return them, and clear the thread_stats request records
  ///
  void TestSwapRequestRecords()
  {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    using ns = std::chrono::nanoseconds;
    auto request_record1 = RequestRecord(
        time_point(ns(1)), std::vector<time_point>{time_point(ns(2))}, 0, false,
        0, false);
    auto request_record2 = RequestRecord(
        time_point(ns(3)), std::vector<time_point>{time_point(ns(4))}, 0, false,
        0, false);
    auto request_record3 = RequestRecord(
        time_point(ns(5)), std::vector<time_point>{time_point(ns(6))}, 0, false,
        0, false);

    std::vector<RequestRecord> source_request_records;

    SUBCASE("No threads")
    {
      auto ret = SwapRequestRecords(source_request_records);
      CHECK(source_request_records.size() == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Source has request records")
    {
      // Any request records in the vector passed in to SwapRequestRecords will
      // be dropped on the floor
      //
      source_request_records.push_back(request_record1);
      auto ret = SwapRequestRecords(source_request_records);
      CHECK(source_request_records.size() == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("One thread")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_records_.push_back(request_record1);
      stat1->request_records_.push_back(request_record2);
      stat1->request_records_.push_back(request_record3);
      threads_stat_.push_back(stat1);

      CHECK(stat1->request_records_.size() == 3);
      auto ret = SwapRequestRecords(source_request_records);
      CHECK(stat1->request_records_.size() == 0);

      REQUIRE(source_request_records.size() == 3);
      CHECK(source_request_records[0] == request_record1);
      CHECK(source_request_records[1] == request_record2);
      CHECK(source_request_records[2] == request_record3);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Multiple threads")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_records_.push_back(request_record2);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->request_records_.push_back(request_record1);
      stat2->request_records_.push_back(request_record3);

      threads_stat_.push_back(stat1);
      threads_stat_.push_back(stat2);

      CHECK(stat1->request_records_.size() == 1);
      CHECK(stat2->request_records_.size() == 2);
      auto ret = SwapRequestRecords(source_request_records);
      CHECK(stat1->request_records_.size() == 0);
      CHECK(stat2->request_records_.size() == 0);

      REQUIRE(source_request_records.size() == 3);
      CHECK(source_request_records[0] == request_record2);
      CHECK(source_request_records[1] == request_record1);
      CHECK(source_request_records[2] == request_record3);
      CHECK(ret.IsOk() == true);
    }
  }

  /// Test the public function GetAccumulatedClientStat
  ///
  /// It will accumulate all contexts_stat data from all threads_stat
  ///
  void TestGetAccumulatedClientStat()
  {
    cb::InferStat result_stat;

    SUBCASE("No threads")
    {
      auto ret = GetAccumulatedClientStat(&result_stat);
      CHECK(result_stat.completed_request_count == 0);
      CHECK(result_stat.cumulative_total_request_time_ns == 0);
      CHECK(result_stat.cumulative_send_time_ns == 0);
      CHECK(result_stat.cumulative_receive_time_ns == 0);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("One thread one context stat")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_[0].completed_request_count = 2;
      stat1->contexts_stat_[0].cumulative_total_request_time_ns = 3;
      stat1->contexts_stat_[0].cumulative_send_time_ns = 4;
      stat1->contexts_stat_[0].cumulative_receive_time_ns = 5;
      threads_stat_.push_back(stat1);

      auto ret = GetAccumulatedClientStat(&result_stat);
      CHECK(result_stat.completed_request_count == 2);
      CHECK(result_stat.cumulative_total_request_time_ns == 3);
      CHECK(result_stat.cumulative_send_time_ns == 4);
      CHECK(result_stat.cumulative_receive_time_ns == 5);
      CHECK(ret.IsOk() == true);
    }
    SUBCASE("Multiple thread multiple contexts")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_.push_back(cb::InferStat());
      stat1->contexts_stat_[0].completed_request_count = 2;
      stat1->contexts_stat_[0].cumulative_total_request_time_ns = 3;
      stat1->contexts_stat_[0].cumulative_send_time_ns = 4;
      stat1->contexts_stat_[0].cumulative_receive_time_ns = 5;
      stat1->contexts_stat_[1].completed_request_count = 3;
      stat1->contexts_stat_[1].cumulative_total_request_time_ns = 4;
      stat1->contexts_stat_[1].cumulative_send_time_ns = 5;
      stat1->contexts_stat_[1].cumulative_receive_time_ns = 6;
      threads_stat_.push_back(stat1);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->contexts_stat_.push_back(cb::InferStat());
      stat2->contexts_stat_.push_back(cb::InferStat());
      stat2->contexts_stat_[0].completed_request_count = 7;
      stat2->contexts_stat_[0].cumulative_total_request_time_ns = 8;
      stat2->contexts_stat_[0].cumulative_send_time_ns = 9;
      stat2->contexts_stat_[0].cumulative_receive_time_ns = 10;
      stat2->contexts_stat_[1].completed_request_count = 11;
      stat2->contexts_stat_[1].cumulative_total_request_time_ns = 12;
      stat2->contexts_stat_[1].cumulative_send_time_ns = 13;
      stat2->contexts_stat_[1].cumulative_receive_time_ns = 14;
      threads_stat_.push_back(stat2);

      auto ret = GetAccumulatedClientStat(&result_stat);
      // 2 + 3 + 7 + 11
      //
      CHECK(result_stat.completed_request_count == 23);
      // 3 + 4 + 8 + 12
      //
      CHECK(result_stat.cumulative_total_request_time_ns == 27);
      // 4 + 5 + 9 + 13
      //
      CHECK(result_stat.cumulative_send_time_ns == 31);
      // 5 + 6 + 10 + 14
      //
      CHECK(result_stat.cumulative_receive_time_ns == 35);

      CHECK(ret.IsOk() == true);
    }
  }

  /// Test the public function CountCollectedRequests
  ///
  /// It will count all request records in the thread_stats (and not modify
  /// the thread_stats in any way)
  ///
  void TestCountCollectedRequests()
  {
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
    using ns = std::chrono::nanoseconds;
    auto request_record1 = RequestRecord(
        time_point(ns(1)), std::vector<time_point>{time_point(ns(2))}, 0, false,
        0, false);
    auto request_record2 = RequestRecord(
        time_point(ns(3)), std::vector<time_point>{time_point(ns(4))}, 0, false,
        0, false);
    auto request_record3 = RequestRecord(
        time_point(ns(5)), std::vector<time_point>{time_point(ns(6))}, 0, false,
        0, false);

    SUBCASE("No threads")
    {
      CHECK(CountCollectedRequests() == 0);
    }
    SUBCASE("One thread")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_records_.push_back(request_record1);
      stat1->request_records_.push_back(request_record2);
      stat1->request_records_.push_back(request_record3);
      threads_stat_.push_back(stat1);

      CHECK(stat1->request_records_.size() == 3);
      CHECK(CountCollectedRequests() == 3);
      CHECK(stat1->request_records_.size() == 3);
    }
    SUBCASE("Multiple threads")
    {
      auto stat1 = std::make_shared<ThreadStat>();
      stat1->request_records_.push_back(request_record2);

      auto stat2 = std::make_shared<ThreadStat>();
      stat2->request_records_.push_back(request_record1);
      stat2->request_records_.push_back(request_record3);

      threads_stat_.push_back(stat1);
      threads_stat_.push_back(stat2);

      CHECK(stat1->request_records_.size() == 1);
      CHECK(stat2->request_records_.size() == 2);
      CHECK(CountCollectedRequests() == 3);
      CHECK(stat1->request_records_.size() == 1);
      CHECK(stat2->request_records_.size() == 2);
    }
  }

  void TestIdle()
  {
    auto stat1 = std::make_shared<ThreadStat>();
    auto stat2 = std::make_shared<ThreadStat>();
    threads_stat_.push_back(stat1);
    threads_stat_.push_back(stat2);

    SUBCASE("All active")
    {
      // If multiple threads are active, their idle times are averaged
      stat1->idle_timer.idle_ns_ = 5;
      stat2->idle_timer.idle_ns_ = 7;
      CHECK(GetIdleTime() == 6);
      ResetIdleTime();
      CHECK(GetIdleTime() == 0);
    }

    SUBCASE("One inactive")
    {
      // If a thread has no idle time, it is considered inactive and not
      // factored in to the average
      stat1->idle_timer.idle_ns_ = 0;
      stat2->idle_timer.idle_ns_ = 7;
      CHECK(GetIdleTime() == 7);
      ResetIdleTime();
      CHECK(GetIdleTime() == 0);
    }
  }
};

TEST_CASE("load_manager_check_health: Test the public function CheckHealth()")
{
  TestLoadManager tlm(PerfAnalyzerParameters{});
  tlm.TestCheckHealth();
}

TEST_CASE(
    "load_manager_swap_request_records: Test the public function "
    "SwapRequestRecords()")
{
  TestLoadManager tlm(PerfAnalyzerParameters{});
  tlm.TestSwapRequestRecords();
}

TEST_CASE(
    "load_manager_get_accumulated_client_stat: Test the public function "
    "GetAccumulatedClientStat()")
{
  TestLoadManager tlm(PerfAnalyzerParameters{});
  tlm.TestGetAccumulatedClientStat();
}

TEST_CASE(
    "load_manager_count_collected_requests: Test the public function "
    "CountCollectedRequests()")
{
  TestLoadManager tlm(PerfAnalyzerParameters{});
  tlm.TestCountCollectedRequests();
}

TEST_CASE("load_manager_batch_size: Test the public function BatchSize()")
{
  PerfAnalyzerParameters params;

  SUBCASE("batch size 0")
  {
    params.batch_size = 0;
  }
  SUBCASE("batch size 1")
  {
    params.batch_size = 1;
  }
  SUBCASE("batch size 4")
  {
    params.batch_size = 4;
  }

  TestLoadManager tlm(params);
  CHECK(tlm.BatchSize() == params.batch_size);
}

TEST_CASE("load_manager: Test public idle time functions")
{
  PerfAnalyzerParameters params;
  TestLoadManager tlm(params);
  tlm.TestIdle();
}

TEST_CASE(
    "send_request_rate_load_manager: testing the GetAndResetNumSentRequests "
    "function")
{
  PerfAnalyzerParameters params{};

  TestLoadManager tlm(params);

  std::shared_ptr<ThreadStat> thread_stat_1{std::make_shared<ThreadStat>()};
  std::shared_ptr<ThreadStat> thread_stat_2{std::make_shared<ThreadStat>()};

  std::chrono::steady_clock::time_point start_time{
      std::chrono::steady_clock::time_point::min()};

  thread_stat_1->num_sent_requests_ = 6;
  thread_stat_2->num_sent_requests_ = 5;

  tlm.threads_stat_ = {thread_stat_1, thread_stat_2};

  const size_t result{tlm.GetAndResetNumSentRequests()};

  CHECK(result == 11);
  CHECK(tlm.threads_stat_.size() == 2);
  CHECK(tlm.threads_stat_[0]->num_sent_requests_ == 0);
  CHECK(tlm.threads_stat_[1]->num_sent_requests_ == 0);
}

}}  // namespace triton::perfanalyzer
