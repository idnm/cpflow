def trace_product(u, v):
    return (u * v.conj().T).sum()


def trace2(u, v, angles):
    return jnp.abs(trace_product(u(angles), v)) ** 2


def all_shifts(u, angles, s):
    basis_shifts = jnp.identity(len(angles))
    return vmap(lambda x: u(angles + x))(s * basis_shifts)


@partial(custom_jvp, nondiff_argnums=(0, 1,))
def shift_trace2(u, v, angles):
    return trace2(u, v, angles)


@shift_trace2.defjvp
def shift_trace2_jvp(u, v, primals, tangents):
    angles, = primals
    tangent_angels, = tangents

    tr = trace_product(u(angles), v)
    tr_shifted_vector = all_shifts(lambda a: trace_product(u(a), v), angles, jnp.pi)

    ans = jnp.abs(tr) ** 2
    der = (tr_shifted_vector * tr.conj()).real * tangent_angels

    return ans, der.sum()
