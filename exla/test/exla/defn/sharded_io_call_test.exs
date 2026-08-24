defmodule EXLA.Defn.ShardedIoCallTest do
  use EXLA.Case, async: false

  alias Nx.Mesh

  defp wait_for_second_callback!(deadline_ms) do
    {:messages, messages} = Process.info(self(), :messages)

    if Enum.any?(messages, &match?({:exla_runtime_call, _, _, _}, &1)) do
      :ok
    else
      if System.monotonic_time(:millisecond) >= deadline_ms do
        raise "timed out waiting for the second partition callback"
      end

      Process.sleep(1)
      wait_for_second_callback!(deadline_ms)
    end
  end

  defp assert_device_lock_available(mesh) do
    parent = self()
    tag = make_ref()

    {pid, ref} =
      spawn_monitor(fn ->
        args = [[Nx.tensor([1, 2, 3])]]
        opts = [client: :host, input_shardings: [%{}]]
        [result] = EXLA.shard_jit(&Nx.add(&1, 1), mesh, opts).(args)
        send(parent, {tag, Nx.to_flat_list(result)})
      end)

    assert_receive {^tag, [2, 3, 4]}, 10_000
    assert_receive {:DOWN, ^ref, :process, ^pid, :normal}
  end

  test "emits a shard-local callback with identity sharding" do
    fun = fn x -> Nx.io_call(x, :tap) end
    mesh = %Mesh{name: "mesh", shape: {2}}
    args = [[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]]

    %{mlir_module: mlir} =
      EXLA.to_mlir_module(fun, args,
        mesh: mesh,
        input_shardings: [%{0 => [0]}],
        hooks: %{tap: fn _ -> :ok end}
      )

    assert mlir =~ "stablehlo.custom_call @exla_runtime_callback"

    assert mlir =~
             "sdy.sharding_rule = #sdy.op_sharding_rule<([i], [j])->([j]) " <>
               "{i=#{EXLA.Executable.callback_server_pid_size()}, j=4} " <>
               "need_replication={i}, custom>"
  end

  test "input materialization failure releases the device lock" do
    mesh = %Mesh{name: "mesh", shape: {1}}

    input =
      Nx.tensor([1, 2, 3],
        backend: {EXLA.Backend, client: :no_automatic_transfers_host, device_id: 0}
      )

    error =
      assert_raise ArgumentError, fn ->
        EXLA.shard_jit(&Nx.io_call(&1, :tap), mesh,
          client: :host,
          input_shardings: [%{}]
        ).([[input]])
      end

    assert error.message =~
             "one of the input tensors are allocated on no_automatic_transfers_host #0 (host)"

    assert_device_lock_available(mesh)
  end

  @tag :spmd_callback
  test "executes once with each local shard and passes both through" do
    parent = self()
    mesh = %Mesh{name: "mesh", shape: {2}}

    fun =
      EXLA.shard_jit(&Nx.io_call(&1, :tap), mesh,
        input_shardings: [%{0 => [0]}],
        hooks: %{tap: &send(parent, {:tap, &1})}
      )

    assert [result0, result1] = fun.([[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]])
    assert_equal(result0, Nx.tensor([1, 2]))
    assert_equal(result1, Nx.tensor([3, 4]))

    values =
      for _ <- 1..2 do
        assert_receive {:tap, value}
        assert value.shape == {2}
        Nx.to_flat_list(value)
      end

    assert Enum.sort(values) == [[1, 2], [3, 4]]
    refute_receive {:tap, _}
  end

  @tag :spmd_callback
  @tag :capture_log
  test "replies to every partition when one callback raises" do
    parent = self()
    mesh = %Mesh{name: "mesh", shape: {2}}

    callback = fn value ->
      send(parent, {:callback_invoked, Nx.to_flat_list(value)})
      wait_for_second_callback!(System.monotonic_time(:millisecond) + 5_000)
      send(parent, :second_callback_pending)
      raise "per-partition callback failed"
    end

    fun =
      EXLA.shard_jit(&Nx.io_call(&1, :tap), mesh,
        input_shardings: [%{0 => [0]}],
        hooks: %{tap: callback}
      )

    {pid, ref} =
      spawn_monitor(fn ->
        fun.([[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]])
      end)

    assert_receive {:DOWN, ^ref, :process, ^pid, {%RuntimeError{message: message}, _}}, 10_000
    assert message =~ "per-partition callback failed"
    refute message =~ "timed out waiting"
    assert_receive {:callback_invoked, values}
    assert values in [[1, 2], [3, 4]]
    assert_receive :second_callback_pending
    refute_receive {:callback_invoked, _}
  end

  test "rejects containers" do
    fun = fn x -> Nx.io_call({x, x}, :tap) end
    mesh = %Mesh{name: "mesh", shape: {2}}
    args = [[Nx.tensor([1, 2])], [Nx.tensor([3, 4])]]

    assert_raise ArgumentError, ~r/supports a single tensor/, fn ->
      EXLA.to_mlir_module(fun, args,
        mesh: mesh,
        input_shardings: [%{0 => [0]}],
        hooks: %{tap: fn _ -> :ok end}
      )
    end
  end

  test "rejects vectorized tensors" do
    fun = fn x -> x |> Nx.vectorize(:batch) |> Nx.io_call(:tap) end
    mesh = %Mesh{name: "mesh", shape: {2}}
    args = [[Nx.tensor([[1, 2], [3, 4]])], [Nx.tensor([[5, 6], [7, 8]])]]

    assert_raise ArgumentError, ~r/does not support vectorized tensors/, fn ->
      EXLA.to_mlir_module(fun, args,
        mesh: mesh,
        input_shardings: [%{0 => [0]}],
        hooks: %{tap: fn _ -> :ok end}
      )
    end
  end

  test "rejects runtime_call" do
    fun = fn x ->
      Nx.runtime_call(Nx.template({2}, :s32), x, [], fn value, _opts -> value end)
    end

    mesh = %Mesh{name: "mesh", shape: {2}}
    args = [[Nx.tensor([1])], [Nx.tensor([2])]]

    assert_raise ArgumentError, ~r/Nx.runtime_call is not supported/, fn ->
      EXLA.to_mlir_module(fun, args, mesh: mesh, input_shardings: [%{0 => [0]}])
    end
  end
end
