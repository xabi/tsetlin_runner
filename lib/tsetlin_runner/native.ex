defmodule TsetlinRunner.Native do
  @moduledoc false
  use Rustler, otp_app: :tsetlin_runner, crate: "tsetlin_nif"

  def load_model_nif(_path), do: :erlang.nif_error(:nif_not_loaded)
  def predict_nif(_resource, _packed_bits), do: :erlang.nif_error(:nif_not_loaded)
end
