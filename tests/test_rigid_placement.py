import numpy as np

from auditembodied.l4_lift import rigid_place_poses,_rotation_vector_matrix


def test_rigid_placement_preserves_grasp_geometry_and_starts_at_prior_command():
    positions=np.array([[.08,-.15,.97],[-.04,.15,1.08]])
    rotations=np.stack([_rotation_vector_matrix(np.array([.2,.1,-.05])),
        _rotation_vector_matrix(np.array([-.1,.2,.1]))])
    table=np.array([[.1,-.16,.92],[-.06,.16,.92]])
    poses=list(rigid_place_poses(positions,rotations,table,.004,60))
    assert len(poses)==60
    np.testing.assert_array_equal(poses[0][0],positions)
    np.testing.assert_array_equal(poses[0][1],rotations)
    length=np.linalg.norm(positions[1]-positions[0])
    relative_position=rotations[0].T@(positions[1]-positions[0])
    relative_rotation=rotations[0].T@rotations[1]
    for p,r in poses:
        np.testing.assert_allclose(np.linalg.norm(p[1]-p[0]),length,atol=1e-12)
        np.testing.assert_allclose(r[0].T@(p[1]-p[0]),relative_position,atol=1e-12)
        np.testing.assert_allclose(r[0].T@r[1],relative_rotation,atol=1e-12)
    p,r=poses[-1]
    np.testing.assert_allclose(p.mean(0),table.mean(0)-[0.,0.,.004],atol=1e-12)
    np.testing.assert_allclose((p[1]-p[0])/length,
        (table[1]-table[0])/np.linalg.norm(table[1]-table[0]),atol=1e-12)
